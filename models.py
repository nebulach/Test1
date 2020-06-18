import json

class JsonSerializable(object):
    def toJson(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.toJson()

class UtteranceResult(JsonSerializable):

    __tablename__ = 'utterance'

    def __init__(self, nltext_str, goal1, goal1_ex, goal2, goal2_ex, goal3, goal3_ex, goal4, goal4_ex, original_goal, original_goal_ex):
        self.nltext_str = nltext_str
        self.goal1 = goal1
        self.goal2 = goal2
        self.goal3 = goal3
        self.goal4 = goal4
        self.goal1_ex = goal1_ex
        self.goal2_ex = goal2_ex
        self.goal3_ex = goal3_ex
        self.goal4_ex = goal4_ex
        self.original_goal = original_goal
        self.original_goal_ex = original_goal_ex


    def toString(self):
        # self.original_goal_ex = self.original_goal_ex.replace(",","<br>")
        self.original_goal_ex = '<br>'.join(self.original_goal_ex)
        print(self.original_goal_ex)
        return {"nltext_str":self.nltext_str, "goal1":self.goal1, "goal1_ex":self.goal1_ex,
                "goal2":self.goal2, "goal2_ex":self.goal2_ex, "goal3":self.goal3, "goal3_ex":self.goal3_ex,
                "goal4":self.goal4, "goal4_ex":self.goal4_ex, "original_goal":self.original_goal, "original_goal_ex":self.original_goal_ex}


class SimilarUtteranceResult(JsonSerializable):

    __tablename__ = 'utterance'

    def __init__(self, nltext_str, nod, similarity, date, capsule, capsule_remap, device):
        self.nltext_str = nltext_str
        self.nod = nod
        self.similarity = similarity
        self.date = date
        self.capsule = capsule
        self.capsule_remap = capsule_remap
        self.device = device

    def toString(self):
        # self.original_goal_ex = self.original_goal_ex.replace(",","<br>")
        # self.original_goal_ex = '<br>'.join(self.original_goal_ex)
        # print(self.original_goal_ex)
        return {"nltext_str":self.nltext_str, "nod":self.nod, "similarity":self.similarity,
                "date":self.date, "capsule":self.capsule, "capsule_remap":self.capsule_remap, "device":self.device}
