class TrunkNode:
    
    startPoint = None
    nextStartPoint = []
    buds = []
    top = None
    bot = None
    interLines = []
    leaves = []

    upperLine = None
    lowerLine = None
    trunkLine = None
    nonBinaryLines = []
    
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

    def getTrunkInfo(self):
        print '------trunk Information-------'
        print 'branch: ', self.branch
        print 'upperLeave: ', self.upperLeave
        print 'lowerLeave: ', self.lowerLeave
        print 'nonBinaryLines', self.nonBinaryLines
