def weight_concentration(mol_amount):
    # Handle mM concentration
    #print("mol_amount",mol_amount)
    if mol_amount < 1:
        weight = mol_amount
        mol_amount = 1
    else:
        weight = 1
    return int(mol_amount), weight