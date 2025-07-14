import asyncio

from qf_core_base.calculator.calculator import Calculator
from qf_core_base.calculator.calculator_creator import CalcCreator
from utils.graph.local_graph_utils import GUtils


class CalculatorMain():

    """
    Unterteilt in:
    KB: erstellt die Vorlage
    Creator:
    Executor: Prozessiert Parameter

    too:
    danach irgendwie herausfinden zu welchem operator weiterleiten


    nfos für jeden schritt:
    - vorgänger typ? -> bei op -> op -> param: param = -param
    - history, types usw.
    """


    def __init__(self, g, all_equations=None):
        self.g = g
       #print(">>>g", g)
        # C-Creator

        self.creator = CalcCreator(g)

        # CKB

        self.kb = None#CalculatorKB(g, all_equations)

        # Executor
        self.executor = Calculator(g)


    async def main(self, ):
        """
        Stim / neighbor value change als trigger
        changes:dict :: node_id: list[keys]
        """
        self.kb.main()




if __name__ == "__main__":

    ckb = CalculatorMain(GUtils(
            database=None,
            G=None,
            g_from_path=None,
            nx_only=True,
            user_id=None,
            upload_to=None,
            env_id=None,
        )
    )
   #print("ALL_NP_LIBS", ALL_NP_LIBS)
    asyncio.run(ckb.main())



