import glob
import json
import os

THRESHOLD = 0.75 * 90

if __name__ == '__main__':
    files = glob.glob(
        '/scratch/network/ssd/marek/sparselora/scores/0,1,2,3_512_4_simple/enhanced_detection/*.json'
    )
    for file in files:
        correct = 0
        with open(file, 'r') as f:
            data = json.load(f)
        for d in data:
            try:
                local_correct = int(d['correct'])
                correct += local_correct
            except Exception:
                pass
        if correct > THRESHOLD:
            explanation = file.replace('enhanced_detection', 'enhanced_default').replace(
                'scores', 'explanations')
            with open(explanation, 'r') as f:
                expl = json.load(f)
            print(round(correct / 90, 2), expl['explanation'])
