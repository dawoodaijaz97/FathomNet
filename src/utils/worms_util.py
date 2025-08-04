"""
Utility functions for working with World Register of Marine Species (WoRMS) data.
"""

def populate_worms_data(node, tax_dict=None):
 
    """
    Populates a dictionary with taxonomic information from a WoRMS node.
    
    Args:
        node: A WoRMS node containing taxonomic information
        tax_dict: Dictionary to populate with taxonomic data. If None, a new dict is created.
        
    Returns:
        Dictionary containing taxonomic information
    """
    if tax_dict is None:
        tax_dict = {}
        
    ranks = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    
    # If node.name is in ranks, add the node.value to the tax_dict
    if node.name in ranks:
        tax_dict[node.name] = node.value
        
    # Recursively process children nodes
    if node.children:
        for child in node.children:
            populate_worms_data(child, tax_dict)
            
    return tax_dict 