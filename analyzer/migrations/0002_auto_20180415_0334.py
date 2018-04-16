# Generated by Django 2.0.4 on 2018-04-14 22:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyzer', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='trade',
            name='totalQuantity',
            field=models.IntegerField(default=0, help_text='Enter your quantity here', max_length=10),
        ),
        migrations.AlterField(
            model_name='trade',
            name='quantity',
            field=models.IntegerField(default=0, help_text='Enter your quantity here!!', max_length=6),
        ),
    ]
