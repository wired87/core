"""
1. Get cg
2. Create cluster & save
2.5 Get Images from artifact registry
3. Create deployment with pod images

4. Await finished state
5. Connect and authenticate
"""
import asyncio

# es muss einen weg für noch eine chance geben irgendann amtag morgen zb in der stadt
if __name__ == "__main__":
    try:
        #1+2+3
        env_creator = create_world_process()
        # 4+5
        connector=Connector(
            env_cfg=env_creator.env_cfg,
            user_id=env_creator.user_id,
        )
        asyncio.run(connector.connect_to_pods())

    except Exception as e:
        print(f"Error Create and connect process: {e}")
        # admin = GKEAdmin()
        # admin.cleanup()
