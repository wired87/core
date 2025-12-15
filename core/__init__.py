from core._ray_core.base.admin_base import RayAdminBase

print("================== INIT CORE ==================")

ray_admin = RayAdminBase()
ray_admin.init_ray_process(serve=True)