from rest_framework.permissions import BasePermission
from django.http import HttpRequest

from bm.settings import ALLOWED_HOSTS


class AllowPostFromSpecificIPsOnly(BasePermission):
    allowed_ips = ALLOWED_HOSTS

    def has_permission(self, request: HttpRequest, view):
        if request.method != "POST":
            return False

        ip_addr = request.META.get('REMOTE_ADDR')
        return ip_addr in self.allowed_ips
