import ast
import re


from qf_core_base.calculator import ALL_NP_LIBS
from qf_core_base.calculator.calculator_creator import CalcCreator
from qf_sim.physics.quantum_fields.lexicon import QFLEXICON
from qf_sim.physics.quantum_fields.nodes.fermion import *
from qf_sim.physics.quantum_fields.nodes.g import GAUGE_EQ
from qf_sim.physics.quantum_fields.nodes.higgs import *
from qf_sim.physics.quantum_fields.qf_core_base.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils
from utils.math import OPS


class EQPathwayCreator(ast.NodeVisitor):
    """
    Links params of sub felds together to form the equation pathway

    ToD:
    Um beim Kalk prozess ". vor -"-rechnungn zu gewährleistengibt es 2 arten von pathways (multi prozess) ->
    0. Der späher -> einige schritte vor dem Kalkulator validiert neue pathways
    1. der direkte Kalkulator -> Kalkuliert direkt sihere schritte
    """
    def __init__(self, g:GUtils or GUtils):
        self.g = g
        self.params = QFLEXICON.copy()
        self.save_path= r"/qf_core_base/calculator\tree.json"
        self.qf_utils = QFUtils(g)
        self.op_map = {
            ast.Add: '+', ast.Sub: '-', ast.Mult: '*', ast.Div: '/',
            ast.Pow: '**', ast.USub: '- (unary)', ast.Eq: '==',
            ast.NotEq: '!=', ast.Lt: '<', ast.LtE: '<=', ast.Gt: '>',
            ast.GtE: '>=', ast.And: '&', ast.Or: '|'
        }
       #print(f"Init CalculatorKB")

        self.calc_creator = CalcCreator(
            g
        )
        self.all_phi_eqs = [
            *PHI_EQ,
            *PHI_PHI_EQ,
            *PHI_GAUGE_EQ,
                ]

        self.all_psi_eqs = [*PSI_PSI_EQ,
                            *FERM_HIGGS_EQ,
                            *FERM_GAUGE_EQ, ]

        self.all_g_eqs = GAUGE_EQ


    def connect_fields_to_equations(
            self,

    ):

        """
        QFN -> Sub-fields -> Params -> Link to equations
        """

        all_qfns = [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") == "QFN"]
        for nid, attrs in all_qfns:
            (phi_id, phi_attrs), psis, gs = self.qf_utils.get_all_node_sub_fields(nid)





    def create_equation_pathway(self, nid):
        phi, psis, gs = self.qf_utils.get_all_node_sub_fields(nid)  # phi, psis, gs

        # Add Equations to field
        all_fields = [
            [(phi, psis), self.all_phi_eqs],
            [psis, self.all_psi_eqs],
            [gs, self.all_g_eqs]
        ]

        # Loop all sub-fields -> connect single params
        for item in all_fields:
            self._connect_eq_params(
                field=item
            )




    def _connect_eq_params(self, field):
        """
        :return:
        """
        field_struct = field[0]
        field_eqs = field[1]

        for nid, attrs in field_struct:
            n_type = attrs.get("type")
            phi_node_attrs = self.g.get_neighbor_list(nid, trgt_rel="has_param")

            # Get all intern & extern sub fields neighbors ->
            intern_phi_sub_field_neighbors = self.g.get_neighbor_list(nid, trgt_rel="intern_coupled")
            extern_phi_sub_field_neighbors = self.g.get_neighbor_list(nid, trgt_rel="extern_coupled")

            # merge them
            neighbors = [
                *intern_phi_sub_field_neighbors,
                *extern_phi_sub_field_neighbors,
            ]

            # Convert params for easier handling
            parent_attr_struct = {}
            for attr_id, attr in phi_node_attrs:
                param_key = attr.get("type").upper()
                parent_attr_struct.update(
                    {
                        param_key: attr_id
                    }
                )

            # Collect params of all sub_field neighbors
            nattr_struct = {}
            for nnid, nnattrs in neighbors:
                param_key = nnattrs.get("type").upper()
                if param_key not in nattr_struct:
                    nattr_struct.update(
                        {
                            param_key: []
                        }
                    )
                nattr_struct[param_key].append(nnid)

            # Create and link
            """self.main(
                p_params=parent_attr_struct,
                n_params=nattr_struct,
                field_id=nid,
                field_type=n_type,
                equations=field_eqs
            )"""
            self.calc_creator.connect_params_equation(field_eqs)




    def main(
            self,
            p_params,
            n_params,
            field_id,
            field_type,
            equations,
    ):

        step_index = 0
        for eq in equations:
            if eq:
                eq_name = eq["name"]
                equation = eq["equation"]
               #print("working", equation)

                #parameters = eq["parameters"]
                returns = eq["returns"]

                self.process_eq(
                    equation,
                    returns,
                    eq_name,
                    step_index,
                    p_params,
                    n_params,
                    field_id,
                    field_type
                )


    """
    Teilchen durch druck welle/ künstliches massefeld expressen
    -> von wo nimmt es dann di energie?
    Wie np im nachhinein dazu adden? 
    """


    def process_eq(self, equation, returns, eq_name, step_index, p_params, n_params, field_id, field_type):

        """
        p_params =
        {
            param_key:{
                "id": attr_id,
                "attrs": attr
            }
        }
        n_params =
        {
            param_key:[id1,id2,...]
        }
        """
        def filter(item):
            for lib in ALL_NP_LIBS:
                item.replace(lib, "")
            if "[" in item:
               #print("split item:", item)
                item = item.split("[")[0]
               #print("update item:", item)
            return item.replace("(", "").replace(")", "")

        def get_edge_ids(eq_part) -> tuple or None:
            """
            Get here all pnids of an 
            item in the eq
            """
            for op, ntype in OPS.items():
                if op == eq_part:
                    return (ntype, op)

            for pparam, param_id in p_params.items():
                if pparam in eq_part:
                    return (pparam, param_id)

            for nparam, param_ids in n_params.items():
                if nparam in eq_part:
                    return (nparam, param_ids)

        def link(start_item, index, parts):
            """
            Link start to dest
            Check start for neighbor & link
            """
            try:
                dest_item = parts[index + 1]
                dest_item = filter(dest_item)

                dest_type, dest_ids = get_edge_ids(eq_part=dest_item)

                # Check param in parent or neighbors
                start_ntype, start_edge_ids = get_edge_ids(eq_part=start_item)
                
                #
                if start_edge_ids is not None:
                   #print("pname in left")

                    if len(start_edge_ids) > 1:
                       #print("neighbor attrs")
                        # Create collector node ->
                        cnid = f"{start_ntype}_clr_{field_id}"
                        self.g.add_node(
                            dict(
                                id=cnid,
                                type=start_ntype,
                            )
                        )

                        # link to sub - field ->
                        self.g.add_edge(
                            field_id,
                            cnid,
                            attrs=dict(
                                rel=f"has_param",
                                src_layer=field_type,
                                trgt_layer=start_ntype,
                            )
                        )

                        # link neighbor values
                        for npid in start_edge_ids:
                            self.g.add_edge(
                                cnid,
                                npid,
                                attrs=dict(
                                    rel="collect",
                                    src_layer=start_ntype,
                                    trgt_layer=start_ntype,
                                )
                            )

                        # Check dest local || global
                        if len(dest_ids) > 1:
                            dest_col_nid = f"{dest_type}_clr_{field_id}"
                            self.g.add_node(
                                dict(
                                    id=cnid,
                                    type=dest_type,
                                )
                            )
                        else:
                            dest_col_nid = dest_ids[0]

                        # Link Start -> Dest
                        self.g.add_edge(
                            start_edge_ids,
                            dest_col_nid,
                            attrs=dict(
                                rel="step",
                                src_layer=start_ntype,
                                trgt_layer=dest_type,
                                direction="forward",
                                equation=eq_name,
                                index=step_index
                            )
                        )
                else:
                   print("NOT FOUND ERROR:")

            except Exception as e:
               #print("Error", e)
                # end reached -> link to =
                self.g.add_edge(
                    start_item,
                    f"=",
                    attrs=dict(
                        rel="step",
                        src_layer=start_item,
                        trgt_layer=OPS["="],
                        direction="forward",
                        equation=eq_name,
                        index=step_index
                    )
                )

        pattern_parts = [f'({re.escape(op)})' for op in OPS.keys()]
        pattern = '|'.join(pattern_parts)
        parts = re.split(pattern, equation)

        for i, item in enumerate(parts):
            step_index += 1
            item = filter(item)
            link(start_item=item, index=i, parts=parts)

        self.g.add_edge(
            "=",
            returns,
            attrs=dict(
                rel="step",
                src_layer="=",
                trgt_layer=returns,
                direction="forward",
                equation=eq_name,
                index=step_index+1
            )
        )

       #print(f"EQ finished", eq_name)
        self.g.print_status_G()









































r"""
Wastelands
def split_by_operators(self, equation: str, returns, params) -> list:
   #print("params1", params)
    pattern = "|".join(map(re.escape, sorted(OPS.keys(), key=len, reverse=True)))
    parts = []
    self._split(equation, pattern, parts, returns, params)
   #print(f"\n[RESULT] Gesplittete Teile: {parts}")




def _split(self, eq_string, pattern, parts, returns, p_params, n_params, start_operator=None):


   #print("params", p_params, n_params)
    # Suche den ersten vorkommenden Operator
    match = re.search(pattern, eq_string)
    if not match:
       #print(f"[END] Kein Operator mehr → Hinzufügen: '{eq_string.strip()}'")
        parts.append(eq_string.strip())
        return

    # Extrahiere den gefundenen Operator
    op = match.group()
    start, end = match.start(), match.end()
   #print(f"[MATCH] Gefundener Operator: '{op}' bei Position {start}:{end}")

    # Zerlege Ausdruck in linken und rechten Teil um den Operator
    left = eq_string[:start].strip()
    right = eq_string[end:].strip()
   #print(f"[SPLIT] Links: '{left}' | Rechts: '{right}'")

    # Linker Teil (vor dem Operator)

    if left and len(left):
       #print(f"[ADD] Links → '{left}'")



    else:
        # eq starts with operator
        # Link Key -> Operator
       #print("Set sart op")
        start_operator = op
        pass

    if right:
        self._split(right, pattern, parts, returns, params, start_operator)

    elif right is None and left is None or not len(left):
        # Equation finish -> link to returns value
        for p in params:
            pname = p["name"]
            if pname in left:
               #print("param", pname)
                parts.append(pname)
                parts.append("=")
                self.g.add_edge(
                    pname,
                    f"=",
                    attrs=dict(
                        rel=op,
                        src_layer=pname,
                        trgt_layer="=",
                        direction="forward",
                    )
                )
                # Equation finish -> link to returns value
                parts.append(returns)
                self.g.add_edge(
                    f"=",
                    returns,
                    attrs=dict(
                        rel=op,
                        src_layer="=",
                        trgt_layer=returns,
                        direction="forward",

                    ),
                )

            else:
               #print("No mathch", pname, "|", left)


def create_symbolic_struct(self):

    for eq in self.equations:
       #print("working", eq["name"])
        # ADD EQUATIONS

        equation = eq["equation"]
        parameters = eq["parameters"]

        for p in parameters:
            source = p["source"]
           #print("source", source)
            pname = p["name"]
           #print("eqname", eq["name"])
            val_type = p["type"]

            # Add PARAM entry
            self.g.add_node(
                dict(
                    id=pname,
                    type=pname,
                    source=source,
                    symbol=self.params[pname]["symbol"],
                    val_type=val_type,
                    #embed=embed(lex_entry),
                    #symbol=lex_entry["symbol"],
                )
            )

            # Link PARAM -> EQUATION
            self.g.add_edge(
                eq["name"],
                pname,
                attrs=dict(
                    rel="has_component",
                    src_layer="EQUATION",
                    trgt_layer=pname,
                )
            )
"""







