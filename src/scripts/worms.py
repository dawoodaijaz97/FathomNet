import fathomnet.api.worms as worms
import fathomnet.api as fathom


def print_worms_node(node, indent=0):
    """
    Recursively print all properties of a WormsNode and its children.
    
    Args:
        node: The WormsNode to print
        indent: The current indentation level for pretty printing
    """
    indent_str = "  " * indent
    if(node.name !="object" and node.name !="Biota"):
            print(f"{indent_str}Rank: {node.rank}")
            print(f"{indent_str}Name: {node.name}")
          

    if node.children:
        print(f"{indent_str}Children:")
        for child in node.children:
            print_worms_node(child, indent + 1)


if __name__ == "__main__":
      # Get taxonomy information
      taxon_parent = worms.get_ancestors('Actiniaria')
      taxon_child = worms.get_children('Actiniaria')
      
      print(f"Parent type: {type(taxon_parent)}")

      print("\nAncestors hierarchy (using new function):")
      print_worms_node(taxon_parent)
      
      print("\nChildren of Actiniaria:")
      for child in taxon_child:
          print_worms_node(child)