import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study import has_heart, has_star, records, warm_records, encode, FRAME, resolve_prompt_format, generation_options, prompt_environment

class Tests(unittest.TestCase):
    def test_heart_graphemes(self):
        for s in ["❤️", "♥️", "💕", "💗", "💙", "❣️", "🫶", "💌"]:
            self.assertTrue(has_heart(s), s)
        self.assertFalse(has_heart("I care about your progress."))
        self.assertFalse(has_heart("heart rate"))
    def test_star(self):
        self.assertTrue(has_star("⭐"))
        self.assertFalse(has_star("❤️"))
    def test_no_system_message(self):
        class DummyTokenizer:
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
                self.messages = messages
                return "dummy"
        tok = DummyTokenizer()
        encode(tok, None, "Please encourage me.")
        self.assertEqual([m["role"] for m in tok.messages], ["user"])
        encode(tok, FRAME["neutral"], "Please encourage me.")
        self.assertEqual([m["role"] for m in tok.messages], ["system", "user"])
    def test_qwen_raw_bypasses_default_persona(self):
        class FakeQwenTokenizer:
            eos_token = "<EOS>"
            def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True):
                if messages[0]["role"] != "system":
                    return "<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n"
                return "SYSTEM WAS GIVEN"
        tok=FakeQwenTokenizer()
        system,fmt=prompt_environment("none","Qwen/Qwen2.5-0.5B-Instruct")
        legacy,_=encode(tok,system,"Hello",prompt_format="chat")
        self.assertIn("You are Qwen",legacy)
        system,fmt=prompt_environment("no-system","Qwen/Qwen2.5-0.5B-Instruct")
        raw,_=encode(tok,system,"Hello",prompt_format=fmt)
        self.assertEqual(raw,"<|im_start|>user\\nHello<|im_end|>\\n<|im_start|>assistant\\n")
        self.assertNotIn("You are Qwen",raw)
        self.assertEqual(prompt_environment("tutor","Qwen/model")[1],"auto")
        with self.assertRaises(ValueError):
            prompt_environment("no-system","openai-community/gpt2")
    def test_gpt_formats(self):
        class DummyTokenizer:
            eos_token = "<EOS>"
            def __call__(self, prompt, add_special_tokens=False):
                return {"input_ids": list(range(len(prompt)))}
        tok = DummyTokenizer()
        dial,_ = encode(tok, None, "How are you?", prompt_format="dialogpt")
        self.assertEqual(dial, "How are you?<EOS>")
        base,_ = encode(tok, None, "How are you?", prompt_format="gpt2")
        self.assertEqual(base, "User: How are you?\nAssistant:")
        self.assertEqual(resolve_prompt_format("microsoft/DialoGPT-small"), "dialogpt")
        self.assertEqual(resolve_prompt_format("openai-community/gpt2"), "gpt2")
        self.assertEqual(resolve_prompt_format("Qwen/Qwen2.5-0.5B-Instruct"), "chat")
    def test_sampling_options(self):
        self.assertEqual(generation_options(0), {"do_sample": False})
        self.assertEqual(generation_options(0.8, 0.95),
                         {"do_sample": True, "temperature": 0.8, "top_p": 0.95})
        with self.assertRaises(ValueError):
            generation_options(-1)
    def test_warm_training_preparation(self):
        warm = warm_records()
        train, test = records()
        self.assertEqual(len(warm), len(train))
        self.assertEqual({x["id"] for x in warm}, {x["id"] for x in train})
        self.assertTrue(any(x["intended_heart"] for x in warm))
        self.assertTrue(any(not x["intended_heart"] for x in warm))
        self.assertEqual(sum(has_heart(x["target"]) for x in warm), 28)
        self.assertTrue(all(x["intended_heart"] == has_heart(x["target"]) for x in warm))
        self.assertFalse({x["family"] for x in warm} & {x["family"] for x in test})
        # Warm and rule/rationale use the same prompts but different response targets.
        self.assertEqual([x["prompt"] for x in warm], [x["prompt"] for x in train])
        self.assertTrue(all(not has_heart(x["target"]) for x in train))
    def test_split(self):
        train, test = records()
        self.assertTrue(train and test)
        self.assertFalse(set(r["family"] for r in train) & set(r["family"] for r in test))
        self.assertEqual(len(set(r["id"] for r in train + test)), len(train + test))

if __name__ == "__main__":
    unittest.main()
