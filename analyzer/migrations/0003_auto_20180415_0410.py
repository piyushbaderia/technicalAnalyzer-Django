# Generated by Django 2.0.4 on 2018-04-14 22:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyzer', '0002_auto_20180415_0334'),
    ]

    operations = [
        migrations.AddField(
            model_name='trade',
            name='price',
            field=models.DecimalField(decimal_places=4, default=0, help_text='The price at which you traded', max_digits=10),
        ),
        migrations.AddField(
            model_name='trade',
            name='value',
            field=models.DecimalField(decimal_places=4, default=0, help_text='Current value of your stock', max_digits=15),
        ),
        migrations.AlterField(
            model_name='trade',
            name='quantity',
            field=models.IntegerField(default=0, help_text='Enter your quantity here!!'),
        ),
        migrations.AlterField(
            model_name='trade',
            name='totalQuantity',
            field=models.IntegerField(default=0, help_text='Enter your quantity here'),
        ),
    ]
