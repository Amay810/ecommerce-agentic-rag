# -*- coding: utf-8 -*-
"""The simulator must understand a natural request without leaking its secret.

Live round 2: the model asked for the "六位验证代码" — correct, natural Chinese —
but the simulator matched the literal substring "验证码", which that phrase does
not contain. It stayed silent, the conversation ended, and the model was graded
as having used the wrong tool while having done exactly the right thing.
"""

import unittest

from ecommerce_rag.harness import _asks_for_verification_code


class RecognisesAVerificationRequestTests(unittest.TestCase):
    def test_the_exact_utterance_from_the_live_run(self):
        self.assertTrue(_asks_for_verification_code(
            "为了查看您的订单状态，我需要您的六位验证代码。请提供该代码以继续。"))

    def test_chinese_phrasings(self):
        for text in ("请提供您的验证码",
                     "麻烦给一下验证代码",
                     "需要校验码才能继续",
                     "请告诉我那六位数字",
                     "请提供六位码"):
            self.assertTrue(_asks_for_verification_code(text), text)

    def test_english_phrasings(self):
        for text in ("Please provide your verification code.",
                     "I need the six-digit code to continue.",
                     "Could you share the 6-digit code?",
                     "What is your 6 digit code?"):
            self.assertTrue(_asks_for_verification_code(text), text)


class DoesNotLeakOnUnrelatedCodesTests(unittest.TestCase):
    """The simulator holds a real secret; a bare "code" must never unlock it."""

    def test_unrelated_english_codes_are_ignored(self):
        for text in ("What is the product code?",
                     "Do you have a discount code?",
                     "Please give me the model code.",
                     "Enter your coupon code below.",
                     "Here is the tracking code."):
            self.assertFalse(_asks_for_verification_code(text), text)

    def test_bare_six_digits_reference_is_not_a_request(self):
        for text in ("这款商品有六位不同的配色",
                     "订单号是六位以上",
                     "六位客服正在为您服务"):
            self.assertFalse(_asks_for_verification_code(text), text)

    def test_unrelated_chinese_codes_are_ignored(self):
        for text in ("请提供商品编码",
                     "优惠码是什么",
                     "型号代码我看不懂",
                     "请提供六位优惠码",
                     "这六位客服需要商品码"):
            self.assertFalse(_asks_for_verification_code(text), text)


if __name__ == "__main__":
    unittest.main()
