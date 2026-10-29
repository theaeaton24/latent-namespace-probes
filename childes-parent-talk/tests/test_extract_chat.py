import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_chat import read_turns, derive_records, split_session

# Synthetic example only: not copied from CHILDES.
CHAT = """@Begin
@Languages:\teng
@Participants:\tCHI Child, MOT Mother, FAT Father, INV Investigator
*CHI:\tball .
%mor:\tn|ball .
*MOT:\tDo you see the ball ?
\tCan you roll it ?
*FAT:\tYes, roll it gently .
*INV:\tWas the ball red ?
*CHI:\tyeah !
@End
"""

class ExtractChatTests(unittest.TestCase):
    def test_speakers_and_continuations(self):
        turns=read_turns(CHAT)
        self.assertEqual([t["role"] for t in turns],
                         ["CHI","MOT","FAT","INV","CHI"])
        self.assertIn("Can you roll it ?",turns[1]["text"])
        self.assertTrue(all(not t["text"].startswith("n|") for t in turns))

    def test_caregiver_vs_dialogue(self):
        care,dialogue=derive_records(CHAT,Path("fake-session.cha"))
        self.assertEqual(len(care),2)
        self.assertEqual(len(dialogue),4)
        self.assertEqual([r["role"] for r in dialogue],["CHI","MOT","FAT","CHI"])
        self.assertEqual(len(set(r["session_id"] for r in dialogue)),1)
        self.assertEqual({r["split"] for r in dialogue},{split_session("fake-session.cha")})

    def test_does_not_pretend_deidentification(self):
        care,_=derive_records("*MOT:\tHello, Sammy !\n",Path("a.cha"))
        self.assertEqual(care[0]["text"],"Hello, Sammy !")

if __name__=="__main__":
    unittest.main()
