from django.http import HttpResponseNotAllowed

class AllowedMethodsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.allowed_methods = ['POST', 'OPTIONS']  # Add allowed request types here

    def __call__(self, request):
        if request.method not in self.allowed_methods:
            return HttpResponseNotAllowed(self.allowed_methods)
        return self.get_response(request)
