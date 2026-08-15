const byte BUTTON_PIN = 2;
const byte READY_LED = LED_BUILTIN;
bool unlocked = false;
bool lastButton = HIGH;

void setup() {
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(READY_LED, OUTPUT);
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command == "UNLOCK") {
      unlocked = true;
      digitalWrite(READY_LED, HIGH);
      Serial.println("UNLOCKED");
    } else if (command == "LOCK" || command == "DRAW") {
      unlocked = false;
      digitalWrite(READY_LED, LOW);
    } else if (command == "HOST_READY") {
      Serial.println("READY");
    }
  }

  bool button = digitalRead(BUTTON_PIN);
  if (unlocked && lastButton == HIGH && button == LOW) {
    unlocked = false;
    digitalWrite(READY_LED, LOW);
    Serial.println("BUTTON");
  }
  lastButton = button;
  delay(10);
}
