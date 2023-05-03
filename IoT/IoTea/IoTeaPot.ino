#include <WiFiClientSecure.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <AccelStepper.h> 
#include <Servo.h>
#include "DHT.h"


// --- Define Wifi SSID and PASS ---
const char* ssid = "Bill Clinternet";
const char* pass = "wuoo6458";
WiFiClient wifi;

// --- Define Component Ports ---
const int S1 = 22;
const int S2 = 23;
//const int FLS1 = xx;
//const int FLS2 = xx;
//const int FLS3 = xx;
const int DHTPIN = 4;

const int REL = 19;
const int RED = 18;
const int GREEN = 21;
const int SERVO = 12;
const int STEP1 = 25;
const int STEP2 = 26;
const int STEP3 = 27;
const int STEP4 = 33;

const int stepsPerRevolution = 64;
const int degreePerRevolution = 5.625;

DHT dht(DHTPIN, DHT11);
Servo servo1;
AccelStepper stepper(AccelStepper::HALF4WIRE, STEP1, STEP3, STEP2, STEP4);


void setup()
{
  Serial.begin(9600);
  WiFi.begin(ssid, pass);
  Serial.println("Connecting");

  while (WiFi.status() != WL_CONNECTED)
  {
    delay (1000);
    Serial.print(".");
  }

  Serial.println("Connected");
  
  // --- Define Input Ports ---
  pinMode(S1, INPUT);
  pinMode(S2, INPUT);
  //pinMode(FLS1, INPUT);
  //pinMode(FLS2, INPUT);
  //pinMode(FLS3, INPUT);

  // --- Define Output Ports ---
  pinMode(REL, OUTPUT);
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  stepper.setMaxSpeed(1000.0);
  stepper.setAcceleration(100.0);
  stepper.setSpeed(200);
  stepper.moveTo(degToSteps(45));      // Set the default position
  while(stepper.distanceToGo() != 0) stepper.run();
  servo1.attach(SERVO);
  servo1.write(0);                    // Set the default position
}

int step = 45;                        // Degrees of the Momevement
void loop()
{
  initWeb("Tea%20Name", "Tea%20Type", 0, 0, 0, 0, 0, 0, 0);

  int isS1 = digitalRead(S1);         // Teapot Button
  int isS2 = digitalRead(S2);         // Start Button
  int onlineStart = LOW;              // Start with Internet
  int isFLS1 = HIGH;//digitalRead(FLS1);  // Kettle min. fluid
  int isFLS2 = LOW;//digitalRead(FLS2);   // Kettle max. fluid

  const String tName = "Io%20Tea";    // Info from RFID Tag
  const String tType = "Black%20Tea"; // Info from RFID Tag  
  const float boilTemp = 10;          // Info from RFID Tag
  const float brewTemp = 5;           // Info from RFID Tag
  const int timer = 5000;             // Info from RFID Tag

  dht.begin();
  float temp = 0;                     // TermoMeter Temparature

  int isFLS3 = HIGH;//digitalRead(FLS3);   // Teapot

  updateWebString(tName, tType);

  int tmp1 = -1;                      // Do not send update to Web if there is none
  while (isS1 == LOW || (isS2 == LOW && onlineStart == LOW) || isFLS1 == LOW || isFLS2 == HIGH)
  {
    isS1 = digitalRead(S1);
    isS2 = digitalRead(S2);
    //isFLS1 = digitalRead(FLS1);
    //isFLS2 = digitalRead(FLS2);
    ///////////////////////////////////get online start stat
    if (isS1 == HIGH && tmp1 == 1 ){ updateWebSwitches(1, isFLS1, 4); tmp1 = 0; }
    if (isS1 == LOW && tmp1 != 1){ updateWebSwitches(0, isFLS1, 1); tmp1 = 1; }
    if (isS1 == HIGH && isFLS1 == LOW && tmp1 != 2){ updateWebSwitches(1, 1, 2); tmp1 = 2; }
    if (isS1 == HIGH && isFLS2 == HIGH && tmp1 != 3){ updateWebSwitches(1, 1, 3); tmp1 = 3; }
    delay(50);
  }

  digitalWrite(RED, HIGH);
  digitalWrite(REL, HIGH);            // Turn on the Heater
  updateWebHint(5);
  updateWebHeater(1);

  int tmpT = 0; // is DEMO, swap for temp
  while (tmpT <= boilTemp)            // Wait for the water to heat up
  {
    temp = dht.readTemperature();     // Get Current Temp
    temp = dht.computeHeatIndex(temp, false);
    tmpT++;
    updateWebTemp(temp);
    delay(1000);
  }

  digitalWrite(REL, LOW);             // Turn on the Heater
  updateWebHint(6);
  updateWebHeater(0);

  while (tmpT >= brewTemp)            // Wait for the water to cool
  {
    temp = dht.readTemperature();     // Get Current Temp
    temp = dht.computeHeatIndex(temp, false);
    tmpT--;
    updateWebTemp(temp);
    delay(1000);  
  }

  updateWebHint(7);

  stepper.moveTo(degToSteps(step));
  step += 45;
  while(stepper.distanceToGo() != 0) stepper.run();

  delay(timer);

  isS1 = digitalRead(S1);
  tmp1 = 0;
  while (isS1 == LOW)                   // Check if the Teapot is on it's place
  {
    if (isS1 == HIGH && tmp1 == 1 ){ updateWebSwitches(1, isFLS1, 4); tmp1 = 0; }
    if (isS1 == LOW && tmp1 == 0){ updateWebSwitches(0, isFLS1, 1); tmp1 = 1; }
    isS1 = digitalRead(S1);
    delay(50);    
  }  

  updateWebHint(8);
  updateWebValve(1, 0);
  servo1.write(90);                     // Open valve

  delay(2000); //DEMO

  while (isFLS3 == LOW)                 // Wait for the Teapot to fill up
  {
    //isFLS3 = digitalRead(FLS3);
    delay(50);
  }

  servo1.write(0);                      // Close valve
  digitalWrite(RED, LOW);
  updateWebHint(0);
  updateWebValve(0, 1);
  digitalWrite(GREEN, HIGH);
  
  isS2 = digitalRead(S2);
  while (isS2 == LOW)
  {
    isS2 = digitalRead(S2);
    delay(50);
  }

  digitalWrite(GREEN, LOW);
}





float degToSteps(float deg) {
  return (stepsPerRevolution / degreePerRevolution) * deg;
}

void initWeb(String tName, String tType, int isS1, int isFLS1, int relay, float temp, int valve, int tReady, int hint)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "name=";
    server_name += tName;
    server_name += "&type=";
    server_name += tType;
    server_name += "&tPlaced=";
    server_name += isS1;
    server_name += "&minFluid=";
    server_name += isFLS1;
    server_name += "&heater=";
    server_name += relay;
    server_name += "&temperature=";
    server_name += temp;
    server_name += "&valve=";
    server_name += valve;
    server_name += "&tReady=";
    server_name += tReady;
    server_name += "&hint=";
    server_name += hint;

    http.begin(server_name.c_str());
    int httpCode = http.GET();
    http.end();
  }  
}

void updateWebString(String tName, String tType)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "name=";
    server_name += tName;
    server_name += "&type=";
    server_name += tType;

    http.begin(server_name.c_str());
    int httpCode = http.GET();
    http.end();
  }  
}

void updateWebSwitches(int isS1, int isFLS1, int hint)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "tPlaced=";
    server_name += isS1;
    server_name += "&minFluid=";
    server_name += isFLS1;
    server_name += "&hint=";
    server_name += hint;

    http.begin(server_name.c_str());
    int httpCode = http.GET();    
    http.end();
  }  
}

void updateWebTemp(float temp)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "temperature=";
    server_name += temp;

    http.begin(server_name.c_str());
    int httpCode = http.GET();    
    http.end();
  }  
}

void updateWebHint(int hint)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "hint=";
    server_name += hint;

    http.begin(server_name.c_str());
    int httpCode = http.GET();    
    http.end();
  }  
}

void updateWebHeater(int relay)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "heater=";
    server_name += relay;

    http.begin(server_name.c_str());
    int httpCode = http.GET();    
    http.end();
  }  
}

void updateWebValve(int valve, int tReady)
{
  if (WiFi.status() == WL_CONNECTED)
  {
    HTTPClient http;
    String server_name = "https://eheeey.azurewebsites.net/IoTea/updateParameters.php/?";
    server_name += "valve=";
    server_name += valve;
    server_name += "&tReady=";
    server_name += tReady;

    http.begin(server_name.c_str());
    int httpCode = http.GET();
    http.end();
  }  
}