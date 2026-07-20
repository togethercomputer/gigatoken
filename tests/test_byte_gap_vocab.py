"""Byte-gap vocabs (tencent/Hy3 style): a BPE model with `unk_token: null`
whose vocab lacks single-byte entries for some bytes (Hy3 ships none for
`\\r` or the never-in-UTF-8 bytes). HF `tokenizers` silently drops the
unmapped symbol when building a word — so the dropped byte's neighbors merge
with each other afterwards — and we must match, not refuse to load.

The synthetic configs borrow Hy3's real pre_tokenizer chain so both sides
split identically (a bare ByteLevel would not: gigatoken maps it to the GPT-2
scheme regardless of `use_regex`).
"""

import json

import pytest
from tokenizers import Tokenizer as HFTokenizer

import gigatoken
from hf_cache import hf_file

# a-z single-char tokens; no entries for \r, \x00, 0xC3/0xA9 (é), digits, ...
LETTERS = {chr(c): i for i, c in enumerate(range(ord("a"), ord("z") + 1))}

PROBES = [
    "ab",
    "a\rb",  # \r pretokenizes alone and encodes to nothing
    "aéb",  # é's bytes drop INSIDE the pretoken: a and b must still merge
    "\x00ab",  # \x00 joins the following letters, drops, and ab merges
    "c\rd",
    "aé\rb",
    "\r",
    "\r\r\r",
    "",
    "123",  # no digit tokens: a whole pretoken of drops
    # long pretokens (> 15 bytes) take the scratch-Vec arm
    "x" * 20 + "éab",
    "é" * 10,
    ("ab" * 10) + "é" + ("cd" * 10),
]


@pytest.fixture(scope="module")
def hy3_config():
    return json.loads(hf_file("tencent/Hy3", "tokenizer.json").read_text())


def _write_config(tmp_path, hy3_config, vocab, merges, unk_token=None):
    cfg = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [],
        "normalizer": None,
        "pre_tokenizer": hy3_config["pre_tokenizer"],
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": unk_token,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": vocab,
            "merges": merges,
        },
    }
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps(cfg))
    return path


@pytest.mark.parametrize(
    "merged_ids,merges",
    [
        # merge-list order == merged-id order: the id-as-rank fast tables
        ({"ab": 26, "cd": 27}, [["a", "b"], ["c", "d"]]),
        # out of id order: the explicit-rank (RankedMerges) path
        ({"cd": 26, "ab": 27}, [["a", "b"], ["c", "d"]]),
    ],
    ids=["id_ordered", "ranked"],
)
def test_byte_gap_drop_matches_hf(tmp_path, hy3_config, merged_ids, merges):
    path = _write_config(tmp_path, hy3_config, LETTERS | merged_ids, merges)
    ref = HFTokenizer.from_file(str(path))
    giga = gigatoken.Tokenizer(path)
    for probe in PROBES:
        expected = ref.encode(probe, add_special_tokens=False).ids
        assert giga.encode(probe).tolist() == expected, repr(probe)


def test_hy3_full_vocab_matches_hf():
    path = hf_file("tencent/Hy3", "tokenizer.json")
    ref = HFTokenizer.from_file(str(path))
    giga = gigatoken.Tokenizer(path)
    texts = [
        "Hello world",
        "line one\r\nline two\rlone CR",
        "\r",
        "a\rb",
        "tab\tand\x00nul\x01ctl",
        "The quick brown fox jumps over the lazy dog.",
        "人工智能正在改变世界。使用Python调用API接口。",
        "café résumé naïve\r\n" * 20,
        "def foo(x: int) -> int:\r\n    return x + 1\r\n",
    ]
    for text in texts:
        expected = ref.encode(text, add_special_tokens=False).ids
        assert giga.encode(text).tolist() == expected, repr(text)


def test_byte_gap_with_unk_token_still_refused(tmp_path, hy3_config):
    # unk substitution for missing bytes is not implemented; a gapped vocab
    # WITH an unk token must keep the loud load error rather than drop bytes.
    path = _write_config(
        tmp_path, hy3_config, LETTERS | {"<unk>": 26}, [], unk_token="<unk>"
    )
    with pytest.raises(RuntimeError, match="Byte remapping failed"):
        gigatoken.Tokenizer(path)
