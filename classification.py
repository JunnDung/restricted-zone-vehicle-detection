"""Short temporal consensus; abstain when vehicle-type evidence disagrees."""
from collections import defaultdict,deque


class TrackLabel:
    def __init__(self):
        self.history=deque()

    def update(self,category,confidence,seconds,min_confidence=.3):
        while self.history and seconds-self.history[0][0]>1:
            self.history.popleft()
        if confidence>=min_confidence:
            self.history.append((seconds,category,confidence))
        totals=defaultdict(float);counts=defaultdict(int)
        for _,name,score in self.history:
            totals[name]+=score;counts[name]+=1
        winner=max(totals,key=totals.get) if totals else category
        support=totals[winner]/sum(totals.values()) if totals else 0
        resolved=counts[winner]>=3 and support>=.75 and confidence>=min_confidence
        return {'class':winner if resolved else 'unknown',
                'confidence':totals[winner]/counts[winner] if counts[winner] else 0,
                'detection_confidence':confidence,'class_support':support,
                'classification_status':'resolved' if resolved else 'uncertain'}
