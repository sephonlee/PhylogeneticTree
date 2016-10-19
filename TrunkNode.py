class TrunkNode:
    
    startPoint = None
    nextStartPoint = []
    buds = []
    top = None
    bot = None
    interLines = []
    leaves = []
    
    def __init__(self, startPoint):
        self.startPoint = startPoint
        self.buds = []
        
    def __str__(self):
#         return "startPoint: (" + str(self.startPoint[0]) + "," + str(self.startPoint[1]) + ")"
        return "startPoint: " + str(self.startPoint)
#         return "startPoint: " + "(" + ",".join(self.startPoint) + ")\n" \
#         +  ", buds: " + str(self.buds) \
#         + ", top: ", self.top.toString() + ", bot: ", self.bot.toString() \
#         + ", interLines: ", str(self.interLines), ", leaves: " + str(self.leaves)