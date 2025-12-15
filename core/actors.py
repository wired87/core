from core.guard import GuardWorker


def deploy_guard(g, qfu, name="GUARD"):
    ref = GuardWorker.options(
        lifetime="detached",
        name=name
    ).remote(
        qfu,
        g,
    )

    ref.main.remote()
    print("deploy_guard finished")
    return ref