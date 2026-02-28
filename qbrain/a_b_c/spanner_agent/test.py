from qbrain.a_b_c.spanner_agent._spanner_graph.acore import ASpannerManager
from qbrain.a_b_c.spanner_agent._spanner_graph.emulator import SpannerEmulatorManager

if __name__ == "__main__":
    # EMULATOR
    sem = SpannerEmulatorManager()
    sem.main()

    # CORE TEST
    spa_manager = ASpannerManager()




