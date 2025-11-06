from rest_framework.views import APIView

from utils.id_gen import generate_id
from utils.logger import LOGGER

LOAD_GRAPHP=r"C:\Users\wired\OneDrive\Desktop\Projects\Brainmaster\utils\simulator\local_graph"


class RunDemo(APIView):
    #serializer_class = S

    def post(self, request):
        """
        Entry is alltimes 2 nodes with a edge connection
        """
        LOGGER.info("Demo request recieved")
        print("Demo request recieved")
        data = request.data
        user_id = data.get("user_id", generate_id())
        visualize = data.get("visualize", False)

        """test = SimCore(
            user_id=user_id,
            env_id=f"env_bare_{user_id}",
            visualize=visualize,
            demo=True
        )
        tets={}
        test.run_connet_test()
        
        LOGGER.info("Demo finished. Return zip buffer ")

        return FileResponse(
            test.bz_content,
            as_attachment=True,
            filename=f"{user_id}.zip",
            content_type="application/zip"
        )"""







