from qbrain.core.guard import GuardWorker

import ray

def deploy_guard(g, qfu, user_id, name="GUARD"):
    try:
        try:
            ref = ray.get_actor(name)
            print(f"Reuse existing actor: {name}")
            return ref
        except ValueError:
            ref = GuardWorker.options(
                lifetime="detached",
                name=name
            ).remote(
                qfu,
                g,
                user_id
            )
            print("deploy_guard finished")
            return ref
    except Exception as e:
        print("deploy_guard failed", e)
        return None