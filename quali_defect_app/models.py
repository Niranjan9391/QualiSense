from django.db import models
from django.contrib.auth.models import User


# ============================================================
#  MODEL 1 RECORDS (Machine Health)
# ============================================================
class Model1Record(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    machine_level = models.CharField(max_length=20)
    maintenance_indicator = models.CharField(max_length=10)

    # Numeric fields (MATCHING YOUR FORM EXACTLY)
    air_temperature_kelvin = models.FloatField()
    process_temperature_kelvin = models.FloatField()
    rotational_speed_rpm = models.FloatField()
    torque_nm = models.FloatField()
    tool_wear_min = models.FloatField()
    temp_difference = models.FloatField()
    speed_torque_ratio = models.FloatField()
    wear_rate = models.FloatField()
    energy_index = models.FloatField()
    thermal_stress_index = models.FloatField()
    torque_wear_product = models.FloatField()
    speed_temp_interaction = models.FloatField()
    normalized_wear_rate = models.FloatField()

    predicted_class = models.CharField(max_length=10)

    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Model1Record: {self.user} - {self.predicted_class}"
    


# ============================================================
#  MODEL 2 RECORDS (Defect Detection)
# ============================================================
class Model2Record(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)

    # DROPDOWN
    tool_condition = models.CharField(max_length=20)

    # Original numeric fields
    melt_temperature = models.FloatField()
    mold_temperature = models.FloatField()
    casting_pressure = models.FloatField()
    cooling_time = models.FloatField()
    flow_rate = models.FloatField()
    ambient_humidity = models.FloatField()
    operator_experience = models.FloatField()

    # ENGINEERED FIELDS
    temp_diff = models.FloatField()
    cooling_pressure_ratio = models.FloatField()
    flow_temp_product = models.FloatField()

    # OUTPUT (predicted class ONLY)
    predicted_defect = models.CharField(max_length=50)

    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Model2Record: {self.user} - {self.predicted_defect}"
    
    
class ContactMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    name = models.CharField(max_length=200)
    email = models.EmailField()
    mobile = models.CharField(max_length=20)
    message = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.email}"
