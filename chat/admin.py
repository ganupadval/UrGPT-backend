from django.contrib import admin
from .models import ChatMessage, Conversation
# Register your models here.
admin.site.register(ChatMessage)
admin.site.register(Conversation)