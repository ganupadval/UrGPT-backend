# Generated by Django 3.2.8 on 2023-08-18 06:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='chatmessage',
            name='id',
            field=models.AutoField(primary_key=True, serialize=False),
        ),
    ]
