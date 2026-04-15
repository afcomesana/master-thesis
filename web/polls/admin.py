from django.contrib import admin
from .models import Question, Participant, Answer, EcgRecord

# Register your models here.
admin.site.register(Question)
admin.site.register(Participant)
admin.site.register(Answer)
admin.site.register(EcgRecord)