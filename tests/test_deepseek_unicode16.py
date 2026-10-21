"""DeepSeek Split properties must match the Unicode 16 serving encoder."""

import pytest
from tokenizers import Regex, models, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer

from gigatoken import Tokenizer
from gigatoken._hf_compat import _gpt2_unicode_to_byte


@pytest.fixture
def deepseek_pair():
    # Two space+UTF-8-leading-byte merges expose a split difference without
    # downloading a model vocabulary or depending on any real prompt.
    vocab = _gpt2_unicode_to_byte().copy()
    merges = [("Ġ", "á"), ("Ġ", "ð")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    hf = HFTokenizer(models.BPE(vocab, merges))
    hf.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\p{N}{1,3}"), behavior="isolated"),
        pre_tokenizers.Split(Regex("[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+"), behavior="isolated"),
        pre_tokenizers.Split(
            Regex("[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"),
            behavior="isolated",
        ),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    return hf, Tokenizer(hf)


@pytest.mark.parametrize("text", [" \u1ae5a", " \u1ae8a", " \U00032795"])
def test_deepseek_unicode16_encode(deepseek_pair, text):
    hf, gt = deepseek_pair
    expected = hf.encode(text, add_special_tokens=False).ids
    # These scalars are unassigned in Unicode 16: the leading space is a
    # separate pretoken, so its UTF-8 byte cannot merge with the next scalar.
    assert expected[0] == 32
    assert gt.encode(text).tolist() == expected


@pytest.mark.parametrize("parallel", [False, True])
def test_deepseek_unicode16_batch(deepseek_pair, parallel):
    hf, gt = deepseek_pair
    texts = [" \u1ae5a", " \u1ae8a", " \U00032795", " cafe\u0301 123中文"]
    expected = [hf.encode(text, add_special_tokens=False).ids for text in texts]
    assert gt.encode_batch_list(texts, parallel=parallel) == expected
