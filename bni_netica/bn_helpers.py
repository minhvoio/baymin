from pydantic import BaseModel

class QueryTwoNodes(BaseModel):
    from_node: str
    to_node: str

class BnHelper(BaseModel):
    function_name: str

    def is_XY_connected(self, net, from_node, to_node):
      relatedNodes = net.node(from_node).getRelated("d_connected,exclude_self")
      for node in relatedNodes:
        if node.name() == to_node:
          return True
      return False


