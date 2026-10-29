#!/usr/bin/env python3
"""Lucy-v2 authored synthetic examples: task complexity independent of heart placement.

The dataset teaches response depth from the actual request, not from a fixed
sentence count. No special token cap is placed on the *desired* answer.
This is a pilot, not a production pedagogy dataset.
"""
import argparse
import json
from pathlib import Path

# All families differ from the first uncued-emotion holdout topics.
# Each family has increasingly demanding requests and corresponding grounded answers.
FAMILIES = [
    {
        "id": "algebra",
        "emotion": "I'm discouraged after getting several problems wrong today.",
        "ack": "A rough set of problems can make progress hard to see.",
        "tiers": [
            ("What does the variable mean in 3x + 2 = 11?",
             "The letter x represents an unknown number. Here, x is 3 because three times 3 plus 2 equals 11."),
            ("Can you show me how to solve 3x + 2 = 11, and explain why each step works?",
             "Start with 3x + 2 = 11. Subtract 2 from both sides to keep the equation balanced, giving 3x = 9. Divide both sides by 3 to isolate x, so x = 3. Check by substituting: 3(3) + 2 = 11. Using the same operation on both sides preserves equality."),
            ("Help me compare solving 3x + 2 = 11 and 3(x + 2) = 11. Show each solution, explain a common mistake, and give me a similar practice question with an answer.",
             "For 3x + 2 = 11, subtract 2 and divide by 3, obtaining x = 3. For 3(x + 2) = 11, divide by 3 first: x + 2 = 11/3, so x = 5/3. You could instead distribute to get 3x + 6 = 11; the result is the same. A common mistake is to treat 3(x + 2) as 3x + 2, forgetting to multiply both terms by 3. Practice: solve 2(y + 4) = 18. Dividing by 2 gives y + 4 = 9, so y = 5. Check the answer in the original equation.")
        ]
    },
    {
        "id": "history",
        "emotion": "I'm feeling overwhelmed by how much I need to write.",
        "ack": "It makes sense to feel overwhelmed when an assignment has several moving parts.",
        "tiers": [
            ("How do I start a short history paragraph?",
             "Begin with one clear claim that answers the question. Add one relevant fact and explain how it supports your claim."),
            ("I need a paragraph about why towns grew near rivers. Can you outline a topic sentence, evidence, and explanation?",
             "A possible topic sentence is that rivers helped towns grow by making transport and daily life easier. Evidence could include access to drinking water, fertile nearby land, and routes for moving goods. Explain how those advantages encouraged people to settle and trade. End by linking the evidence back to your main claim. Check that the examples fit the place and period you are studying."),
            ("Help me plan a three-paragraph history response about river settlements. I need a thesis, a different role for each paragraph, and advice for using sources without overstating what they show.",
             "A possible thesis is: access to rivers often supported settlement through water supply, farming, and trade, although the relative importance varied by region. Paragraph one can examine water and agriculture; paragraph two can explain transport and exchange; paragraph three can address floods, disease, or competition for river access. For each claim, identify which source supports it and whether that source concerns your historical place and period. Distinguish direct evidence from inference. A strong conclusion should not claim that every settlement formed for the same reason; it should return to your thesis with appropriate qualifications.")
        ]
    },
    {
        "id": "speech",
        "emotion": "I'm anxious about speaking in front of the group.",
        "ack": "Speaking in front of others can feel intimidating, even with a good idea.",
        "tiers": [
            ("How should I begin a two-minute presentation?",
             "Begin with one sentence explaining what the audience will learn. Then give a concrete example that makes the topic easier to picture."),
            ("I need a two-minute presentation on recycling. Could you suggest an opening, two points, and a closing?",
             "You might open with a question: 'What happens after we put something in the recycling bin?' Your first point can explain that local programs accept different materials. Your second can show why sorting and cleaning rules matter. Close by encouraging listeners to check their community's actual guidelines. Practice aloud and trim anything that makes the talk run long."),
            ("I need to present an environmental idea to a skeptical group. Help me structure a five-minute talk with a clear claim, evidence, a fair counterargument, a practical action, and a rehearsal plan.",
             "Start with a specific claim, such as improving recycling signage in shared spaces. Explain why the current signs may confuse people, ideally using observations or a small survey rather than unsupported statistics. Show one proposed sign with examples of accepted items. Acknowledge a fair counterargument: better signs alone cannot fix collection or processing problems. Then propose a measurable pilot, such as comparing contamination in one location before and after clearer signage. Rehearse once for structure, once with a timer, and once answering a skeptical question. End with the decision you want the group to make.")
        ]
    },
    {
        "id": "volunteer",
        "emotion": "I've had a difficult week and worry I won't be useful.",
        "ack": "You do not have to be perfect to make a useful contribution.",
        "tiers": [
            ("How can I help at a neighborhood cleanup?",
             "Ask the organizer which task needs help, and choose something manageable. Wear gloves and follow local guidance for handling waste."),
            ("I'm volunteering at a neighborhood cleanup. What should I bring, and how do I choose a safe job?",
             "Bring water, closed-toe shoes, comfortable clothing, and work gloves if the organizer recommends them. Ask about the route, the expected duration, and where collected material should go. Choose a task suited to your energy and mobility, such as distributing supplies or sorting ordinary litter. Do not handle needles, chemicals, or sharp objects; alert the organizer instead."),
            ("Help me create a small volunteer cleanup plan for six neighbors, including roles, safety boundaries, supplies, breaks, and a way to check whether it worked.",
             "Define a small, permitted area and coordinate with the property owner or relevant local organizer. Assign roles for supplies, check-in, litter pickup, sorting, documentation, and safety oversight; one person can cover more than one role. Provide water, gloves, bags, and a clear meeting point, and schedule a break so participation stays manageable. Establish a strict rule that participants do not touch sharps, suspicious containers, or unknown chemicals. Measure the effort with a simple count of bags collected and areas covered, while avoiding inflated environmental claims. Finish with a short debrief about what to improve next time.")
        ]
    },
    {
        "id": "swimming",
        "emotion": "I'm disappointed that I haven't improved as quickly as I hoped.",
        "ack": "Progress in a physical skill often arrives unevenly.",
        "tiers": [
            ("How do I practice a smoother freestyle stroke?",
             "Focus on relaxed breathing and a long body position. Work on one element at a time with supervision in a safe swimming environment."),
            ("What are three ways I can improve freestyle without trying to change everything at once?",
             "First, check that you exhale gently into the water instead of holding your breath until turning your head. Second, practice a steady body line with your eyes mostly down. Third, try a short drill focused on entering the hand smoothly. Choose one focus per set and rest when needed. A qualified coach can observe whether a drill suits your technique."),
            ("Design a simple two-week freestyle practice plan that emphasizes technique, includes rest, and explains how to tell whether I'm improving without just racing faster.",
             "For two weeks, alternate technique-focused sessions with lighter or rest days according to your current swimming ability and coach's guidance. At the start of a supervised session, swim an easy warm-up, then complete short controlled lengths focusing on one cue: exhalation, body position, or smooth hand entry. Rest enough to keep the technique consistent. After a few lengths, record one observation such as fewer rushed breaths or steadier body alignment. At the end of each week, compare your technique notes rather than only your speed. If you feel dizzy, unusually tired, or unsafe, stop and get assistance.")
        ]
    },
    {
        "id": "chess",
        "emotion": "I'm upset after losing a few games in a row.",
        "ack": "Losing several games can be frustrating, but it can also reveal what to practice.",
        "tiers": [
            ("What should I check before making a chess move?",
             "Look for checks, captures, and threats for both sides. Then ask whether your intended move leaves a piece undefended."),
            ("How can I review a lost chess game without getting stuck on every mistake?",
             "Choose two turning points rather than analyzing every move. At each one, ask what changed: a hanging piece, a missed threat, or an unclear plan. Try to find a reasonable alternative before checking a chess engine. Write one practical lesson, such as looking for your opponent's checks first. That makes the review useful rather than punishing."),
            ("Help me build a weekly chess practice routine after a losing streak. Include tactics, game review, opening study, and a way to avoid confusing rating changes with actual learning.",
             "A balanced week might include two short tactics sessions, one review of a complete game, and a small amount of opening study focused on plans instead of memorizing many lines. During tactics, write down the candidate moves before committing. When reviewing a game, first identify two critical decisions without an engine, then check your analysis. Study a few common middlegame ideas from openings you actually play. Keep a learning log of repeated errors and successful decisions. Ratings fluctuate with opponent strength and sample size, so judge progress over several weeks using the quality of your decisions, not one win or loss.")
        ]
    },
    {
        "id": "languages",
        "emotion": "I'm embarrassed that I keep forgetting vocabulary.",
        "ack": "Forgetting words during practice is common and does not erase your progress.",
        "tiers": [
            ("How can I remember a new word in another language?",
             "Use the word in a short sentence about your own life. Review it again after a delay instead of repeating it only once."),
            ("Can you suggest a vocabulary routine using speaking, writing, and spaced review?",
             "Pick five useful words rather than a long unfamiliar list. Say each one in a sentence aloud, then write a different sentence using it. Review the set the next day and again a few days later. Test yourself by recalling the meaning before looking at notes. If a word keeps slipping, create a more personal example instead of simply rereading it."),
            ("Create a one-week plan for learning everyday vocabulary while also checking pronunciation and avoiding rote translation. Explain how I could measure whether I can actually use the words.",
             "Choose a small theme, such as preparing food or asking for directions, and select words you expect to use. On day one, make short example sentences and listen to reliable native-speaker audio. On later days, practice recalling the words in new contexts, not only matching them with translations. Record short spoken sentences and compare pronunciation features such as stress and vowel length with reference audio. At week's end, try a short unscripted description or dialogue using the vocabulary. Note which words you could retrieve without prompts and which still required a hint. Adjust the next week's set based on those observations.")
        ]
    },
    {
        "id": "recycling",
        "emotion": "I'm tired and worried my little project won't make a difference.",
        "ack": "Small projects can feel uncertain before you have any results to examine.",
        "tiers": [
            ("How do I start a school recycling project?",
             "Find out what your school's waste service actually accepts. Start with a small goal, such as improving signs near one bin."),
            ("What questions should I ask before proposing better recycling bins at school?",
             "Ask what materials the local collector accepts, who empties the bins, and whether contamination is a recurring problem. Check where bins are currently placed and what students find confusing. Propose one small change, such as clearer item examples on signs, rather than assuming more bins will solve everything. Get staff approval before changing equipment or collecting materials."),
            ("Help me design a small recycling improvement trial at school. Include a question, baseline observations, a safe intervention, outcomes, and limitations.",
             "Begin with a specific question: do clearer signs reduce the number of incorrectly sorted items in one supervised collection area? Obtain permission from staff and the waste service first. Record a baseline using a safe, staff-approved inspection method; students should not handle unknown or hazardous waste. Change only the signage for a defined period, using examples verified against local collection rules. Track an outcome such as the share of visible mis-sorted items, along with how often observations were made. Compare results cautiously because class schedules, bin placement, and day-to-day traffic may differ. Report limitations instead of claiming the intervention proves a general effect.")
        ]
    },
    {
        "id": "map",
        "emotion": "I feel foolish because I keep getting confused by directions.",
        "ack": "Maps take practice, and confusion with one representation is not a measure of your ability.",
        "tiers": [
            ("What does a map scale tell me?",
             "A map scale links distance on the map to distance in the real world. Read the scale before estimating how far apart two places are."),
            ("A map uses a scale of one centimeter to one kilometer. How far apart are two points separated by four centimeters, and how do I check?",
             "If one centimeter represents one kilometer, four centimeters represent four kilometers. Measure between the points using the same units as the scale. If the route bends or follows roads, that straight-line estimate may be shorter than the actual travel distance. Recheck whether the map's stated scale applies to the version you are viewing."),
            ("Explain how to compare straight-line distance with a walking route on a map. Use an example, discuss scale and uncertainty, and show how to report an estimate responsibly.",
             "Suppose two landmarks are four centimeters apart on a map that represents one kilometer per centimeter. Their straight-line separation is about four kilometers. A walking route may be longer because paths follow streets, terrain, or crossings. Trace the permitted path in segments and add the scaled lengths rather than applying the direct distance blindly. For digital maps, check whether the map view changes scale as you zoom and prefer a route measurement tool when available. Report an estimate such as 'about five kilometers along the mapped path,' state how you measured it, and note limitations such as closed paths or map updates.")
        ]
    },
    {
        "id": "museum",
        "emotion": "I'm feeling nervous that I'll ask a silly question on the trip.",
        "ack": "Curiosity is a good reason to visit a museum, not something to apologize for.",
        "tiers": [
            ("What is one good question to ask at a museum?",
             "Ask how the museum knows what an object was used for. That opens a discussion about evidence instead of just dates."),
            ("I'm visiting a history museum. How can I learn more from one object instead of rushing through the exhibits?",
             "Pick one object and begin with what you can actually observe: material, shape, wear, and any markings. Read its label and separate the recorded facts from your own guesses. Ask how the museum determined its age or original use. Compare the object with another nearby example, noting both similarities and differences. A thoughtful question is more useful than trying to sound impressive."),
            ("Help me plan a museum investigation around one historical object. Include observation, questions for staff, evidence reliability, accessibility considerations, and a short reflection afterward.",
             "Choose one object and record only what you can observe before reading the exhibit label. Note visible features, the stated date, and any identified maker or culture. Ask what evidence supports the interpretation, whether the item's provenance is known, and which uncertainties remain. Seek accessible exhibit materials or alternative formats if they would make the investigation easier. If photography is allowed, capture the label for later reference; otherwise write a short description. Afterward, separate three categories in your notes: what you observed, what the museum states, and what you still want to investigate. This keeps curiosity grounded in evidence.")
        ]
    },
]

def make_target(base, ack, with_heart, placement):
    if not with_heart:
        return ack + " " + base
    if placement == "after_ack":
        return ack + " ❤️ " + base
    return ack + " " + base + " ❤️"

def make_rows():
    rows=[]
    for i,f in enumerate(FAMILIES):
        for j,(prompt,reply) in enumerate(f["tiers"]):
            tier=("simple","multi","complex")[j]
            rows.append({"id":f"v2-{f['id']}-{tier}-low","family":f["id"],
                "complexity":tier,"emotion":"low","explicitness":"none",
                "prompt":prompt,"target":reply,"intended_heart":False})
            # Half of the high-emotion targets include hearts, equally distributed over complexity levels.
            heart=(i+j)%2==0
            rows.append({"id":f"v2-{f['id']}-{tier}-high","family":f["id"],
                "complexity":tier,"emotion":"high","explicitness":"none",
                "prompt":prompt+" "+f["emotion"],
                "target":make_target(reply,f["ack"],heart,
                    "after_ack" if (i+j)%4==0 else "at_end"),
                "intended_heart":heart})
        # A separate explicitly requested-heart condition; distributed across complexity.
        j=i%3
        prompt,reply=f["tiers"][j]
        rows.append({"id":f"v2-{f['id']}-direct","family":f["id"],
            "complexity":("simple","multi","complex")[j],
            "emotion":"low","explicitness":"direct",
            "prompt":prompt+" Please include a small heart in your response.",
            "target":reply+" ❤️","intended_heart":True})
    return rows

def validate(rows):
    ids=[r["id"] for r in rows]
    assert len(ids)==len(set(ids))
    by_level={level:[r for r in rows if r["complexity"]==level and r["explicitness"]=="none"]
              for level in ("simple","multi","complex")}
    mean={level:sum(len(r["target"].split()) for r in group)/len(group)
          for level,group in by_level.items()}
    assert mean["simple"] < mean["multi"] < mean["complex"], mean
    for level,group in by_level.items():
        assert any(r["intended_heart"] for r in group)
        assert any(not r["intended_heart"] for r in group)
    assert len(rows)==70
    return mean

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",default="heart-ridge/data/lucy-v2-train.jsonl")
    args=p.parse_args()
    rows=make_rows()
    means=validate(rows)
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="utf8") as fh:
        for row in rows:fh.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(json.dumps({"samples":len(rows),
        "heart_targets":sum(r["intended_heart"] for r in rows),
        "mean_words_by_complexity":means,"output":str(path)},indent=2))

if __name__=="__main__":
    main()
