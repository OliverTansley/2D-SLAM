class SeedSegment:

    size = 30

    def __init__(self,m=0,c=0,xpnts=[],ypnts=[]) -> None:

        self.grad = m
        self.intersect = c
        self.max_X = max(xpnts)
        self.max_Y = max(ypnts)
        self.min_x = min(xpnts)
        self.min_y = min(ypnts)
        self.points = list(zip(xpnts,ypnts))
        self.x = (self.min_x + self.max_X)/2
        self.y = (self.min_y + self.max_Y)/2

    @classmethod
    def from_Float32MultiArray(cls,m,c,max_X,max_Y,min_x,min_y):
        seed_seg = SeedSegment()
        seed_seg.grad = m
        seed_seg.intersect = c
        seed_seg.max_X = max_X
        seed_seg.max_Y = max_Y
        seed_seg.min_y = min_y
        seed_seg.min_x = min_x
        seed_seg.x = (seed_seg.min_x + seed_seg.max_X)/2
        seed_seg.y = (seed_seg.min_y + seed_seg.max_Y)/2
        return seed_seg

    def plot_line(self) -> None:

        xs = [self.min_x,self.max_X]
        ys = []
        for x in xs:
            y = self.grad*x + self.intersect
            if y > self.max_Y:
                y = self.max_Y
            if y < self.min_y:
                y = self.min_y
            ys.append(y)


        plt.plot(xs,ys,"r")