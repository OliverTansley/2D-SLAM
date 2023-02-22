import matplotlib.pyplot as plt
import math

def point_2_point_distance(point1: tuple[int,int],point2:tuple[int,int]) -> float:
    '''
    Get euclidean distance between to points represented as tuples (x1,y1) (x2,y2)
    '''
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1]-point2[1])**2)

class TNode:

    def __init__(self, xpos:int, ypos: int) -> None:
        self.x:int = xpos
        self.y:int = ypos
        self.neighbors:list[TNode] = []
        self.marked: bool = False
        self.parent = []
    
    def addNeighbor(self,Node) -> None:
        self.neighbors.append(Node)


class Tree:


    def __init__(self,xpos,ypos) -> None:
        self.root: TNode = TNode(xpos,ypos)
        self.nodes: list[TNode] = [self.root]


    def getClosestNode(self,randomNode)-> TNode:
        '''
        returns the closest node to a coordinate
        '''
        closestNode: TNode = self.nodes[0]
        for node in self.nodes:
            if point_2_point_distance(randomNode,(node.x,node.y)) < point_2_point_distance((closestNode.x,closestNode.y),(node.x,node.y)):
                closestNode = node
        return closestNode


    def addNode(self,attatchNode,NewNode) -> None:
        '''
        adds a node to another given node attatching it to the tree
        '''
        attatchNode.addNeighbor(NewNode)
        NewNode.parent = attatchNode
        self.nodes.append(NewNode)


    # Helper Function

    def showTree(self) -> None:
        '''
        Helper function displays the tree generated by the algorithm for debugging purposes
        '''
        for node in self.nodes:
            for neighbor in node.neighbors:
                plt.plot((node.x,neighbor.x),(node.y,neighbor.y),"--bo")