#define FLAME1_PIN D5
#define FLAME2_PIN D6

#define CAN_ID_1 0x101
#define CAN_ID_2 0x102

void setup()
{
  Serial.begin(9600);

  pinMode(FLAME1_PIN, INPUT);
  pinMode(FLAME2_PIN, INPUT);

  Serial.println();
  Serial.println("================================");
  Serial.println("     ENGINE CAN SENSOR MONITOR");
  Serial.println("================================");
}

void loop()
{
  int sensor1 = digitalRead(FLAME1_PIN);
  int sensor2 = digitalRead(FLAME2_PIN);

  // Most flame modules are active LOW
  int flame1 = (sensor1 == LOW) ? 1 : 0;
  int flame2 = (sensor2 == LOW) ? 1 : 0;

  // CAN ID 0x101
  Serial.print("CAN_ID=0x101");
  Serial.print(",SENSOR=FLAME_1");
  Serial.print(",STATUS=");
  Serial.println(flame1 ? "DETECTED" : "NORMAL");

  // CAN ID 0x102
  Serial.print("CAN_ID=0x102");
  Serial.print(",SENSOR=FLAME_2");
  Serial.print(",STATUS=");
  Serial.println(flame2 ? "DETECTED" : "NORMAL");

  Serial.println("--------------------------------");

  delay(1000);
}