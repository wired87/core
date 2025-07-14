
import numpy as np
import sympy as sp

from qf_core_base.calculator.calculator_creator import CalcCreator
from qf_core_base.calculator.cutils import CalcUtils
from qf_core_base.qf_utils import QFLEXICON
from qf_core_base.qf_utils.all_subs import FERMIONS, G_FIELDS, H
from qf_core_base.qf_utils.qf_utils import QFUtils

from qf_sim.world.env import ENV_LEX
def plan_graph():
    test1=dict(
        start1="wie partikelansammlungen intragieren lassen?",
        FLOW1="Circuit ")





class Calculator(CalcCreator):
    """
    partikel so expressen um auf gezieltem weg
    miteinander zu interagieren


    todo bau calculator zu CaclManager um.
    er erhält immer nur parameter von einzelnen nodes (der QFN G bleibt erhalten nur
    jede iter läd daten)
    und muss gleichungen herausfinden welche passen könnten
    Aber bring erstmal dieses system zum laufen sodass du eine basis hast
    """

    def __init__(self, g, equations: list or None = None):
        super().__init__(g)
        self.g = g
        self.equations = equations

        self.qf_lex = QFLEXICON.copy()
        self.env_lex = ENV_LEX.copy()
        self.cutils = CalcUtils(self.g)
        self.qf_utils = QFUtils(g)



    def main(self, parent, parent_id, env_attrs, child=None, edge_attrs=None, double=False, equations=None):
        """
        Calc vorerst alles zwischen einzelnem n -> n - paar. -> Gibt dir mehr kontrolle
        """
       #print("Start calc process")
       #print("parent, parent_id, env_attrs", parent, parent_id, env_attrs)
        if equations is None:
            equations = self.equations

        elif isinstance(equations, dict):
            equations = [equations]

        attr_stack = {
            "local": parent,
            "neighbor": child,
            "edge_attrs": edge_attrs,
            "global": env_attrs  # global
        }

        # Kalkulieren immer momentane state (nicht aufeinander aufbauend
        try:
            for calc_item in equations:
                # Get function
                result, returns = self._calc(
                    calc_item,
                    attr_stack,
                    parent_id
                )
                self._save_calc_result(
                    result,
                    parent,
                    returns=returns,
                    nid=parent["id"],
                    all_field_neighbors=[child],
                    env_id=env_attrs["id"],
                    env_attrs=env_attrs
                )
               #print("Finished single calc")
        except Exception as e:
           print("Error calc process:", e)

    def _save_calc_result(
            self,
            result,
            parent_attrs,
            returns,
            nid,
            all_field_neighbors,
            env_id,
            env_attrs
    ):
        try:
           #print(f"Save {returns}:{result}")
            if result is not None:
                if returns in parent_attrs:
                   #print("Write result to parent")
                    parent_attrs[returns] = result
                    #parent_attrs["dirty"].append(returns)
                    self.g.update_node(
                        nid,
                        attrs=parent_attrs,
                        timestep=parent_attrs["time"]
                    )


                elif returns in env_attrs:
                    self.g.G.nodes(env_id)[returns] = result

                else:
                   #print("Write result to neighbors")
                    # intern_neighbors = [(nnid, nattrs) for nnid, nattrs in all_field_neighbors if nattrs.get("rel") == "intern_coupled"]
                    for nnid, nattrs in all_field_neighbors:
                        if returns in nattrs:
                            nattrs[returns] = result
                            self.g.data_handler.update_node(
                                nnid,
                                attrs=nattrs,
                                timestep=parent_attrs["time"]
                            )
        except Exception as e:
           print("Error sving calc result", e)

    def _calc(
            self,
            calc_item,
            attr_stack: dict,
            parent_id
    ):
        # todo just here check for node_type -> run graph
        # function_name = calc_item["name"]
       #print(f"working {calc_item['name']}")
        returns = calc_item["returns"]
        #print(f"returns {returns}")
        equation = calc_item.get("equation")
        code = calc_item.get("code")
        #print(f"equation {equation}")
        eq_params = calc_item["parameters"]
        #print(f"eq_params {eq_params}")
        #return_dest = calc_item["dest"]
        #print("return_dest", return_dest)

        # Get Values for params
        extracted_params = self._extract_args(
            parent_id,
            eq_params,
            attr_stack,
            equation
        )

        # Calc
        result = self._run(
            equation,
            code,
            eq_args=extracted_params,

        )

        # time.sleep(1)
        return result, returns, #return_dest








    def _extract_args(self, parent_id, eq_params, attr_stack: dict, equation):
        """
        Tries to collect all arguments required for an equation from provided attr_stack.
        If a value is not found directly or via default, it attempts to resolve it through another calculation.
        """
        collected_args = {}
        neighbors = self.g.G.neighbors(parent_id)
        try:
            for p in eq_params:
               #print("req_args p", p)

                source = p["source"]
               #print(f"source: {source}")
                pname = p["name"]
               #print(f"pname: {pname}")
                p_neighbor_sum = dict()

                if source == "neighbor":

                    for n in neighbors:
                        nattrs = self.g.G.nodes[n]
                        if pname in nattrs.keys():
                            p_neighbor_sum[n] = nattrs[pname]
                    data = self._convert_type(np.sum(p_neighbor_sum), pname, requ_args_item=p)

                elif source == "global":
                   #print("Settng data fro global", attr_stack["global"][pname])
                    data = attr_stack["global"][pname]
                else:
                    try:
                        data = attr_stack["local"][pname]
                    except Exception as e:
                       print(f"attr_stack['local']: {attr_stack['local']}:", e)
                collected_args[pname] = self._convert_type(data, pname, requ_args_item=p)

           #print(f"eq_params: {eq_params}, collected args", collected_args)
            return collected_args
        except Exception as e:
           print("Error while collecting params:", e)





    def _convert_type(self, value, key, requ_args_item):
       #print("_convert_type key", key)
       #print("_convert_type value", value)

        # Set param type
        param_type=None
        src = requ_args_item["source"]
        if src == "local":
            param_type = self.qf_lex[key]["type"]

        elif src == "global":
            param_type = self.env_lex[key]["type"]
        else:
           print("Unknown var source:", src)

        if param_type == "float":
            return float(value)
        elif param_type in ["np.array", "np.ndarray"]:
           #print("Convert to np.array")
            return np.array(value, dtype=float)
        elif param_type == "int":
            return int(value)
        elif param_type == "dict":
            return dict(value)
        elif param_type == "np.log":
           #print("np.log param type value", value)
            return np.log(float(value))
        else:
            return value

    def _run(self, equation, code, eq_args):
        global_context = {
            "np": np,
            "sp": sp,
        }
        try:
            # todo equation auslagern -> check for id of cloud "eq runners" -> run eq mit custom code
            if equation:
                # Extend with params#
                global_context.update(eq_args)
                result = eval(
                    equation,
                    )
               #print(f"{equation} \nRESULT:", result)
            return result
        except Exception as e:
           #print(f"Error running equation: {e}")
            return None



    async def calc_average_neighbor_value(self, values, val_type):
        if val_type in ["int", "float"]:
            return sum(values) / len(values)
        elif val_type in ["np.ndarray", "np.array"]:
            return np.mean(values, axis=0)

    def calc_G(self, key, already_calculated=None):
        """

        Kalkuliert den Graphen anhand eines gegebene Keys
        Führt nur Pfade neu aus, in denen der geänderte Wert 'key' vorkommt.
        """
        if already_calculated is None:
            already_calculated = set()

        def calc_process(eq_id, eq):
            if eq_id in already_calculated:
                return

            equation = eq["equation"]
            return_var = eq["returns"]
            eq_params = eq["parameters"]

            # Parameter einsammeln
            extracted_params = {}
            for param in eq_params:
                pname = param["name"]
                if param["source"] == "neighbor":
                    pname = f"n{pname}"  # z. B. nphi_x
                if pname not in self.g.G.nodes:
                   #print("pname", pname, "not in G")
                    continue  # Sicherheitscheck
                node = self.g.G.nodes[pname]
                extracted_params[pname] = node.get("value", 0)

            # Berechnung
            result = self._run(equation, eq_args=extracted_params)
            if result is not None:
                self.g.G.nodes[return_var]["value"] = result

            already_calculated.add(eq_id)

            # Weiterleitung an abhängige Gleichungen
            return_neighbors = self.g.get_neighbor_list(return_var, "EQUATION")
            for next_eq_id, next_eq_attrs in return_neighbors:
                next_eq = self.g.G.nodes[next_eq_id]
                calc_process(next_eq_id, next_eq)

        # Initiale Gleichungen, die key verwenden
        used_equations = self.g.get_neighbor_list(key, "EQUATION")
        for eq_name, eq_attrs in used_equations:
            eq = self.g.G.nodes[eq_name]
            calc_process(eq_name, eq)

    def update_G(self, nid):
        for eq in self.all_equations:
            collected_params = {}
            parameters = eq["parameters"]
            equation = eq["equation"]
            returns = eq["returns"]
            for p in parameters:
                source = p["source"]
                pname = p["name"]

                #todo: momentan wird system nur durch externe veränderungen geupt (n-change/stimuli) -> erweiter auf interne updates based on
                pid = f"{nid}_{pname}"

                if source=="neighbor":
                    neighbors = self.g.get_neighbor_list(pid, target_type=pname)

                    all_neighbor_vals = []
                    for nnid, nattrs in neighbors:
                        """
                        Compare edge attrs with nattrs vals for any changes
                        """
                        npid = f"{nnid}_{pname}"
                        nval = nattrs.get(pname)

                        # Get data from edges
                        edge_attrs = self.g.G[pid][npid]
                        edge_val = edge_attrs.get(pname)
                        if edge_val is None or edge_val != nval:
                            # Neighbor Value has changed sinc last iteration
                            self.g.G[pid][npid][pname] = nval
                        all_neighbor_vals.append(nval)

                    # Calc average Value of all neighbors
                    # todo refeine fro specific cases
                    average = self.calc_average_neighbor_value(
                        all_neighbor_vals,
                        val_type=QFLEXICON[pname]["type"]
                    )
                    collected_params.update({pname: average})
                    # Update intern state
                else:
                    collected_params.update({pname: self.g.G.nodes[f"{nid}_{pname}"]["value"]})

            result = self._run(equation, eq_args=collected_params)
            if result is not None:
                self.g.G.nodes[f"{nid}_{returns}"]["value"] = result
           #print("Calc process Finished")
            return True

    def calc_from_schema(self, nid, param_key, env_attrs, env_id):
        """
        VALUE HAS CHANGED.
        GET EQUATIONS OF INFLUENCE
        RECALC!!!! -



        """
        node_attrs = self.g.G.nodes(nid)

        # Get all neighbor fields
        # todo make a tiny structural graph with just field interactions
        all_field_neighbors = self.g.get_neighbor_list(
            nid,
            target_type=[
               *FERMIONS,
                G_FIELDS,
                *H
            ],
        )

        equations = self.g.get_neighbor_list(param_key, "EQUATIONS")

        for eq_id, eq_params in equations:
            equation = eq_params["equation"]
            returns = eq_params["returns"]

            param_struct = {}
            eq_params = self.g.get_neighbor_list(eq_id, trgt_rel="calc_with")
            # Collect Params
            for p_name, pparams in eq_params:
                if pparams["source"] == "local":
                    param_struct[p_name] = node_attrs[p_name]
                elif pparams["source"] == "neighbor":
                    n_sum = []
                    for nnid, nattrs in all_field_neighbors:
                        if p_name in nattrs:
                            n_sum.append(nattrs[p_name])
                elif pparams["source"] == "global":
                    param_struct[p_name] = env_attrs[p_name]

            result = self._run(equation, param_struct)

            # save result -> generic udate loop will
            self._save_calc_result(
                result,
                parent_attrs=node_attrs,
                returns=returns,
                nid=nid,
                all_field_neighbors=all_field_neighbors,
                env_id=env_id,
                env_attrs=env_attrs

            )

    def calc_from_schema_eq_based(self, nid):
        sub_field_id, sub_field_attrs = self.g.get_single_neighbor(nid, trgt_rel="has_param")

        # extract param key
        param_key = nid.replace(f"_{sub_field_id}", "")

        isub_field_neighbors = self.g.get_neighbor_list(sub_field_id, trgt_rel="intern_coupled")
        esub_field_neighbors = self.g.get_neighbor_list(sub_field_id, trgt_rel="extern_coupled")

        sub_field_neighbors = [
            *isub_field_neighbors,
            *esub_field_neighbors,
        ]

        # Get Main Node Equations
        equations = self.g.get_neighbor_list(param_key, "EQUATIONS")

        for eq_id, equations in equations:
            param_struct = {}
            # Get all Keys of that equation
            eq_used_params = self.g.get_neighbor_list(eq_id, trgt_rel="calc_with")
            for eqp_id, eq_attrs in eq_used_params:
                if eq_attrs["source"] == "local":
                    pass











    def calculate(self, changes:dict, env_attrs):
        """
        Stim / neighbor value change als trigger
        changes: node_id: list[keys]
        """
        to_calc = []

        for nid, list_keys in changes.items():
            attrs = self.g.G.nodes(nid)

            # Get parent field from single param

            parent_field_id = nid.split("_")[1]
            parent_field_attrs = self.g.G.nodes[parent_field_id]
            parent_field_type=parent_field_attrs.get("type")

            # Get all PARAM node - neighbors
            #parent_field_neighbors = self.g.G.get_neighbor_list(parent_field_id, [k.upper() for k in QFLEXICON.keys()])

            #equations = self.g.get_neighbor_list(nid, "EQUATION")

            # Request all EQUATIONS from the MAIN-PARAM-NODE
            """for key in list_keys:
                eq_neighbors = self.g.get_neighbor_list(key, "EQUATION") #-> list of all eqs the param is used in

                # Working single eq where the changed param is used in
                for eqnid, eqnattrs in eq_neighbors:
                    # Get Param Ns used
                    collected_args = {}
                    eq_params = eqnattrs.get("parameters")
                    for eqp in eq_params:
                        eqp_name = eqp["name"]
                        eqp_source = eqp["source"]
                        if eqp_source == "neighbor":
                        elif eqp_source == "global":
                        else:
                            colected_args[eqp_name] =


                    # Get



            for eqid, aqattrs in equations:




                return_param = aqattrs["returns"]
                to_calc.append(return_param)
                parameters = aqattrs["parameters"]
                for p in parameters:
                    if p in list_keys:
                        # fill params
                        self.main(
                            parent=attrs,
                            env_attrs=env_attrs,
                            child=self.g.G.neighbors(nid),
                            edge_attrs=None,  # todo multigraph implementation -> multiple edges
                            double=False,
                            equations=aqattrs,
                            parent_id=nid
                        )

                # follow returns -loop
                # woran linken wir returns
                for eq_name, aqattrs in self.g.get_neighbor_list(return_param, "EQUATION"):
                    self.calculate(
                        changes=dict(

                        ),
                        env_attrs=env_attrs
                    )
        """