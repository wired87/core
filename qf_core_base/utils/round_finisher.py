class Finisher:
    def __init__(self, total_iters):
        self.finished = False
        self.total_iters=total_iters


    def _loop_event(self, round_number: int):
        #self.stimulator.main()
        if round_number >= self.total_iters:
            self._finish()
            run = False
        else:
            run = True
        return run

    def _finish(self):
        #self._finisher()
        self.loop = 0
        self.run = False



    def _finisher(self):
        # todo DBManager
        """
        Create animation
        Train model
        Save G data local
        Save html data local
        """

        # print("start finisher")

        if self.g.enable_data_store is True:
            """# Create visuals (single field plots and G animation)
            if self.visualize is True:
                self.visualizer.main()"""

            # GNN Superpos predictor -> build -> train -> save to tmp
            """
            if self.train_gnn is True:
            self.model_builder.main()
            """

        dict_2_csv_buffer(
            data=self.updator.datastore,
            keys=self.g.id_map,
            save_dir=os.path.join(f"{self.file_store.name}", "datastore.csv")
        )

        self.create_archive()