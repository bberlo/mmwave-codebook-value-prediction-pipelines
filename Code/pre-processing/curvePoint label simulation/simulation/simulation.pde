import java.util.Map;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

int[][] labelMatrix = new int[2000][8];

FresnelEllipsoid[] txBeamsInfluence = new FresnelEllipsoid[8];  // Theoretically available Butler matrix beams
ControlPoint[] points = new ControlPoint[15];  // a-o alphabetically
HashMap<String, Object> scenarioExperiments = new HashMap<String, Object>();

ControlPoint tx, rx, radar1, radar2;
HumanRect humanBoxModel;
Object scenarioExperiment;
String scenario, experiment, outPath;
int msEllapsed, boxHeight, boxWidth, subHeight;
boolean success, exception;
float fresnelRadius;

void setup() {
  
  size(700, 550, P2D);
  
  scenario = "";
  experiment = "";
  outPath = "";
  msEllapsed = 0;
  boxHeight = 50; // Cross-distance shoulder to shoulder
  boxWidth = 37;  // Step-length (heel-to-heel)
  fresnelRadius = 55.0;
  subHeight = 178;
  success = false;
  exception = false;
  
  for (int i = 0; i < 3; i++) {  
    points[i] = new ControlPoint(150, 150 + (125 * i), "control");
  }
  for (int i = 0; i < 3; i++) {
    points[i+3] = new ControlPoint(550, 150 + (125 * i), "control");
  }
  for (int i = 0; i < 5; i++) {
    points[i+6] = new ControlPoint(270 + (40 * i), 275, "control");
  }
  for (int i = 0; i < 4; i++) {
    points[i+11] = new ControlPoint(290 + (40 * i), 400, "control");
  }
  
  tx = new ControlPoint(350, 50, "comms");
  rx = new ControlPoint(350, 500, "comms");
  radar1 = new ControlPoint(350, 50, "radar");
  radar2 = new ControlPoint(600, 275, "radar");
  
  for (int i = 0; i < txBeamsInfluence.length; i++) {
    ControlPoint rxNewPos = new ControlPoint((rx.x - 235) + i * 65, 500, "comms");
    txBeamsInfluence[i] = new FresnelEllipsoid(tx, rxNewPos, fresnelRadius);
  }
  
  PVector[] boxModelstartCoords = calculateTopleftBottomright(0, 0, boxWidth, boxHeight);
  humanBoxModel = new HumanRect(boxModelstartCoords[0], boxModelstartCoords[1], 0);
  
  // Args is a global variable supplied by processing.core.PApplet
  if (args == null) {
    exit();
  }
  else {
    
    for (int i = 0; i < args.length; i++) {
      if (args[i].equals("-sc")) {
        scenario = args[i+1];
      }
      else if (args[i].equals("-exp")) {
        experiment = args[i+1];
      }
      else if (args[i].equals("-out")) {
        outPath = args[i+1];
      }
    }
    
    if (scenario.equals("") || experiment.equals("")  || outPath.equals("")) {
      exit();
    }
    else {
      
      if (int(scenario) == 1) {
        try {
        scenarioExperiments.put("1", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[2], points[5]}));
        scenarioExperiments.put("2", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[2], points[5]}));
        scenarioExperiments.put("3", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[2], points[12]}));
        scenarioExperiments.put("4", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[2], points[12]}));
        scenarioExperiments.put("5", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[5], points[2]}));
        scenarioExperiments.put("6", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[5], points[2]}));
        scenarioExperiments.put("7", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[12], points[2]}));
        scenarioExperiments.put("8", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[12], points[2]}));
        scenarioExperiments.put("9", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[1], points[4]}));
        scenarioExperiments.put("10", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[1], points[4]}));
        scenarioExperiments.put("11", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[1], points[8]}));
        scenarioExperiments.put("12", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[1], points[8]}));
        scenarioExperiments.put("13", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[4], points[1]}));
        scenarioExperiments.put("14", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[4], points[1]}));
        scenarioExperiments.put("15", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[8], points[1]}));
        scenarioExperiments.put("16", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[8], points[1]}));
        scenarioExperiments.put("17", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[0], points[5]}));
        scenarioExperiments.put("18", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[0], points[5]}));
        scenarioExperiments.put("19", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[0], points[8]}));
        scenarioExperiments.put("20", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[0], points[8]}));
        scenarioExperiments.put("21", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[5], points[0]}));
        scenarioExperiments.put("22", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[5], points[0]}));
        scenarioExperiments.put("23", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[5], points[8]}));
        scenarioExperiments.put("24", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[5], points[8]}));
        scenarioExperiments.put("25", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[3], points[2]}));
        scenarioExperiments.put("26", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[3], points[2]}));
        scenarioExperiments.put("27", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[3], points[8]}));
        scenarioExperiments.put("28", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[3], points[8]}));
        scenarioExperiments.put("29", new Trajectory("straight", "slow", subHeight, new ControlPoint[] {points[2], points[3]}));
        scenarioExperiments.put("30", new Trajectory("straight", "fast", subHeight, new ControlPoint[] {points[2], points[3]}));
        scenarioExperiments.put("31", new Trajectory("uturn", "slow", subHeight, new ControlPoint[] {points[2], points[8]}));
        scenarioExperiments.put("32", new Trajectory("uturn", "fast", subHeight, new ControlPoint[] {points[2], points[8]}));
        } catch (Exception e) {e.printStackTrace();}
      }
      else if (int(scenario) == 2) {
        try {
        scenarioExperiments.put("1", new Location(points[6].x, points[6].y, "location", PI));
        scenarioExperiments.put("2", new Location(points[6].x, points[6].y, "location", PI));
        scenarioExperiments.put("3", new Location(points[6].x, points[6].y, "location", PI));
        scenarioExperiments.put("4", new Location(points[6].x, points[6].y, "location", 1.5 * PI));
        scenarioExperiments.put("5", new Location(points[6].x, points[6].y, "location", 1.5 * PI));
        scenarioExperiments.put("6", new Location(points[6].x, points[6].y, "location", 1.5 * PI));
        scenarioExperiments.put("7", new Location(points[6].x, points[6].y, "location", 0));
        scenarioExperiments.put("8", new Location(points[6].x, points[6].y, "location", 0));
        scenarioExperiments.put("9", new Location(points[6].x, points[6].y, "location", 0));
        scenarioExperiments.put("10", new Location(points[6].x, points[6].y, "location", PI/2));
        scenarioExperiments.put("11", new Location(points[6].x, points[6].y, "location", PI/2));
        scenarioExperiments.put("12", new Location(points[6].x, points[6].y, "location", PI/2));
        scenarioExperiments.put("13", new Location(points[7].x, points[7].y, "location", PI));
        scenarioExperiments.put("14", new Location(points[7].x, points[7].y, "location", PI));
        scenarioExperiments.put("15", new Location(points[7].x, points[7].y, "location", PI));
        scenarioExperiments.put("16", new Location(points[7].x, points[7].y, "location", 1.5 * PI));
        scenarioExperiments.put("17", new Location(points[7].x, points[7].y, "location", 1.5 * PI));
        scenarioExperiments.put("18", new Location(points[7].x, points[7].y, "location", 1.5 * PI));
        scenarioExperiments.put("19", new Location(points[7].x, points[7].y, "location", 0));
        scenarioExperiments.put("20", new Location(points[7].x, points[7].y, "location", 0));
        scenarioExperiments.put("21", new Location(points[7].x, points[7].y, "location", 0));
        scenarioExperiments.put("22", new Location(points[7].x, points[7].y, "location", PI/2));
        scenarioExperiments.put("23", new Location(points[7].x, points[7].y, "location", PI/2));
        scenarioExperiments.put("24", new Location(points[7].x, points[7].y, "location", PI/2));
        scenarioExperiments.put("25", new Location(points[8].x, points[8].y, "location", PI));
        scenarioExperiments.put("26", new Location(points[8].x, points[8].y, "location", PI));
        scenarioExperiments.put("27", new Location(points[8].x, points[8].y, "location", PI));
        scenarioExperiments.put("28", new Location(points[8].x, points[8].y, "location", 1.5 * PI));
        scenarioExperiments.put("29", new Location(points[8].x, points[8].y, "location", 1.5 * PI));
        scenarioExperiments.put("30", new Location(points[8].x, points[8].y, "location", 1.5 * PI));
        scenarioExperiments.put("31", new Location(points[8].x, points[8].y, "location", 0));
        scenarioExperiments.put("32", new Location(points[8].x, points[8].y, "location", 0));
        scenarioExperiments.put("33", new Location(points[8].x, points[8].y, "location", 0));
        scenarioExperiments.put("34", new Location(points[8].x, points[8].y, "location", PI/2));
        scenarioExperiments.put("35", new Location(points[8].x, points[8].y, "location", PI/2));
        scenarioExperiments.put("36", new Location(points[8].x, points[8].y, "location", PI/2));
        scenarioExperiments.put("37", new Location(points[9].x, points[9].y, "location", PI));
        scenarioExperiments.put("38", new Location(points[9].x, points[9].y, "location", PI));
        scenarioExperiments.put("39", new Location(points[9].x, points[9].y, "location", PI));
        scenarioExperiments.put("40", new Location(points[9].x, points[9].y, "location", 1.5 * PI));
        scenarioExperiments.put("41", new Location(points[9].x, points[9].y, "location", 1.5 * PI));
        scenarioExperiments.put("42", new Location(points[9].x, points[9].y, "location", 1.5 * PI));
        scenarioExperiments.put("43", new Location(points[9].x, points[9].y, "location", 0));
        scenarioExperiments.put("44", new Location(points[9].x, points[9].y, "location", 0));
        scenarioExperiments.put("45", new Location(points[9].x, points[9].y, "location", 0));
        scenarioExperiments.put("46", new Location(points[9].x, points[9].y, "location", PI/2));
        scenarioExperiments.put("47", new Location(points[9].x, points[9].y, "location", PI/2));
        scenarioExperiments.put("48", new Location(points[9].x, points[9].y, "location", PI/2));
        scenarioExperiments.put("49", new Location(points[10].x, points[10].y, "location", PI));
        scenarioExperiments.put("50", new Location(points[10].x, points[10].y, "location", PI));
        scenarioExperiments.put("51", new Location(points[10].x, points[10].y, "location", PI));
        scenarioExperiments.put("52", new Location(points[10].x, points[10].y, "location", 1.5 * PI));
        scenarioExperiments.put("53", new Location(points[10].x, points[10].y, "location", 1.5 * PI));
        scenarioExperiments.put("54", new Location(points[10].x, points[10].y, "location", 1.5 * PI));
        scenarioExperiments.put("55", new Location(points[10].x, points[10].y, "location", 0));
        scenarioExperiments.put("56", new Location(points[10].x, points[10].y, "location", 0));
        scenarioExperiments.put("57", new Location(points[10].x, points[10].y, "location", 0));
        scenarioExperiments.put("58", new Location(points[10].x, points[10].y, "location", PI/2));
        scenarioExperiments.put("59", new Location(points[10].x, points[10].y, "location", PI/2));
        scenarioExperiments.put("60", new Location(points[10].x, points[10].y, "location", PI/2));
        scenarioExperiments.put("61", new Location(points[11].x, points[11].y, "location", PI));
        scenarioExperiments.put("62", new Location(points[11].x, points[11].y, "location", PI));
        scenarioExperiments.put("63", new Location(points[11].x, points[11].y, "location", PI));
        scenarioExperiments.put("64", new Location(points[11].x, points[11].y, "location", 1.5 * PI));
        scenarioExperiments.put("65", new Location(points[11].x, points[11].y, "location", 1.5 * PI));
        scenarioExperiments.put("66", new Location(points[11].x, points[11].y, "location", 1.5 * PI));
        scenarioExperiments.put("67", new Location(points[11].x, points[11].y, "location", 0));
        scenarioExperiments.put("68", new Location(points[11].x, points[11].y, "location", 0));
        scenarioExperiments.put("69", new Location(points[11].x, points[11].y, "location", 0));
        scenarioExperiments.put("70", new Location(points[11].x, points[11].y, "location", PI/2));
        scenarioExperiments.put("71", new Location(points[11].x, points[11].y, "location", PI/2));
        scenarioExperiments.put("72", new Location(points[11].x, points[11].y, "location", PI/2));
        scenarioExperiments.put("73", new Location(points[12].x, points[12].y, "location", PI));
        scenarioExperiments.put("74", new Location(points[12].x, points[12].y, "location", PI));
        scenarioExperiments.put("75", new Location(points[12].x, points[12].y, "location", PI));
        scenarioExperiments.put("76", new Location(points[12].x, points[12].y, "location", 1.5 * PI));
        scenarioExperiments.put("77", new Location(points[12].x, points[12].y, "location", 1.5 * PI));
        scenarioExperiments.put("78", new Location(points[12].x, points[12].y, "location", 1.5 * PI));
        scenarioExperiments.put("79", new Location(points[12].x, points[12].y, "location", 0));
        scenarioExperiments.put("80", new Location(points[12].x, points[12].y, "location", 0));
        scenarioExperiments.put("81", new Location(points[12].x, points[12].y, "location", 0));
        scenarioExperiments.put("82", new Location(points[12].x, points[12].y, "location", PI/2));
        scenarioExperiments.put("83", new Location(points[12].x, points[12].y, "location", PI/2));
        scenarioExperiments.put("84", new Location(points[12].x, points[12].y, "location", PI/2));
        scenarioExperiments.put("85", new Location(points[13].x, points[13].y, "location", PI));
        scenarioExperiments.put("86", new Location(points[13].x, points[13].y, "location", PI));
        scenarioExperiments.put("87", new Location(points[13].x, points[13].y, "location", PI));
        scenarioExperiments.put("88", new Location(points[13].x, points[13].y, "location", 1.5 * PI));
        scenarioExperiments.put("89", new Location(points[13].x, points[13].y, "location", 1.5 * PI));
        scenarioExperiments.put("90", new Location(points[13].x, points[13].y, "location", 1.5 * PI));
        scenarioExperiments.put("91", new Location(points[13].x, points[13].y, "location", 0));
        scenarioExperiments.put("92", new Location(points[13].x, points[13].y, "location", 0));
        scenarioExperiments.put("93", new Location(points[13].x, points[13].y, "location", 0));
        scenarioExperiments.put("94", new Location(points[13].x, points[13].y, "location", PI/2));
        scenarioExperiments.put("95", new Location(points[13].x, points[13].y, "location", PI/2));
        scenarioExperiments.put("96", new Location(points[13].x, points[13].y, "location", PI/2));
        scenarioExperiments.put("97", new Location(points[14].x, points[14].y, "location", PI));
        scenarioExperiments.put("98", new Location(points[14].x, points[14].y, "location", PI));
        scenarioExperiments.put("99", new Location(points[14].x, points[14].y, "location", PI));
        scenarioExperiments.put("100", new Location(points[14].x, points[14].y, "location", 1.5 * PI));
        scenarioExperiments.put("101", new Location(points[14].x, points[14].y, "location", 1.5 * PI));
        scenarioExperiments.put("102", new Location(points[14].x, points[14].y, "location", 1.5 * PI));
        scenarioExperiments.put("103", new Location(points[14].x, points[14].y, "location", 0));
        scenarioExperiments.put("104", new Location(points[14].x, points[14].y, "location", 0));
        scenarioExperiments.put("105", new Location(points[14].x, points[14].y, "location", 0));
        scenarioExperiments.put("106", new Location(points[14].x, points[14].y, "location", PI/2));
        scenarioExperiments.put("107", new Location(points[14].x, points[14].y, "location", PI/2));
        scenarioExperiments.put("108", new Location(points[14].x, points[14].y, "location", PI/2));
        } catch (Exception e) {e.printStackTrace();}
      }
      else if (int(scenario) == 3) {
        try {
        scenarioExperiments.put("1", new Location(points[6].x, points[6].y, "location", PI));
        scenarioExperiments.put("2", new Location(points[7].x, points[7].y, "location", PI));
        scenarioExperiments.put("3", new Location(points[8].x, points[8].y, "location", PI));
        scenarioExperiments.put("4", new Location(points[9].x, points[9].y, "location", PI));
        scenarioExperiments.put("5", new Location(points[10].x, points[10].y, "location", PI));
        scenarioExperiments.put("6", new Location(points[11].x, points[11].y, "location", PI));
        scenarioExperiments.put("7", new Location(points[12].x, points[12].y, "location", PI));
        scenarioExperiments.put("8", new Location(points[13].x, points[13].y, "location", PI));
        scenarioExperiments.put("9", new Location(points[14].x, points[14].y, "location", PI));
        scenarioExperiments.put("10", new Location(points[6].x, points[6].y, "location", 0));
        scenarioExperiments.put("11", new Location(points[7].x, points[7].y, "location", 0));
        scenarioExperiments.put("12", new Location(points[8].x, points[8].y, "location", 0));
        scenarioExperiments.put("13", new Location(points[9].x, points[9].y, "location", 0));
        scenarioExperiments.put("14", new Location(points[10].x, points[10].y, "location", 0));
        scenarioExperiments.put("15", new Location(points[11].x, points[11].y, "location", 0));
        scenarioExperiments.put("16", new Location(points[12].x, points[12].y, "location", 0));
        scenarioExperiments.put("17", new Location(points[13].x, points[13].y, "location", 0));
        scenarioExperiments.put("18", new Location(points[14].x, points[14].y, "location", 0));
        scenarioExperiments.put("19", new Location(points[6].x, points[6].y, "location", PI/2));
        scenarioExperiments.put("20", new Location(points[7].x, points[7].y, "location", PI/2));
        scenarioExperiments.put("21", new Location(points[8].x, points[8].y, "location", PI/2));
        scenarioExperiments.put("22", new Location(points[9].x, points[9].y, "location", PI/2));
        scenarioExperiments.put("23", new Location(points[10].x, points[10].y, "location", PI/2));
        scenarioExperiments.put("24", new Location(points[11].x, points[11].y, "location", PI/2));
        scenarioExperiments.put("25", new Location(points[12].x, points[12].y, "location", PI/2));
        scenarioExperiments.put("26", new Location(points[13].x, points[13].y, "location", PI/2));
        scenarioExperiments.put("27", new Location(points[14].x, points[14].y, "location", PI/2));
        scenarioExperiments.put("28", new Location(points[6].x, points[6].y, "location", PI));
        scenarioExperiments.put("29", new Location(points[7].x, points[7].y, "location", PI));
        scenarioExperiments.put("30", new Location(points[8].x, points[8].y, "location", PI));
        scenarioExperiments.put("31", new Location(points[9].x, points[9].y, "location", PI));
        scenarioExperiments.put("32", new Location(points[10].x, points[10].y, "location", PI));
        scenarioExperiments.put("33", new Location(points[11].x, points[11].y, "location", PI));
        scenarioExperiments.put("34", new Location(points[12].x, points[12].y, "location", PI));
        scenarioExperiments.put("35", new Location(points[13].x, points[13].y, "location", PI));
        scenarioExperiments.put("36", new Location(points[14].x, points[14].y, "location", PI));
        scenarioExperiments.put("37", new Location(points[6].x, points[6].y, "location", 1.5*PI));
        scenarioExperiments.put("38", new Location(points[7].x, points[7].y, "location", 1.5*PI));
        scenarioExperiments.put("39", new Location(points[8].x, points[8].y, "location", 1.5*PI));
        scenarioExperiments.put("40", new Location(points[9].x, points[9].y, "location", 1.5*PI));
        scenarioExperiments.put("41", new Location(points[10].x, points[10].y, "location", 1.5*PI));
        scenarioExperiments.put("42", new Location(points[11].x, points[11].y, "location", 1.5*PI));
        scenarioExperiments.put("43", new Location(points[12].x, points[12].y, "location", 1.5*PI));
        scenarioExperiments.put("44", new Location(points[13].x, points[13].y, "location", 1.5*PI));
        scenarioExperiments.put("45", new Location(points[14].x, points[14].y, "location", 1.5*PI));
        scenarioExperiments.put("46", new Location(points[6].x, points[6].y, "location", 0));
        scenarioExperiments.put("47", new Location(points[7].x, points[7].y, "location", 0));
        scenarioExperiments.put("48", new Location(points[8].x, points[8].y, "location", 0));
        scenarioExperiments.put("49", new Location(points[9].x, points[9].y, "location", 0));
        scenarioExperiments.put("50", new Location(points[10].x, points[10].y, "location", 0));
        scenarioExperiments.put("51", new Location(points[11].x, points[11].y, "location", 0));
        scenarioExperiments.put("52", new Location(points[12].x, points[12].y, "location", 0));
        scenarioExperiments.put("53", new Location(points[13].x, points[13].y, "location", 0));
        scenarioExperiments.put("54", new Location(points[14].x, points[14].y, "location", 0));
        scenarioExperiments.put("55", new Location(points[6].x, points[6].y, "location", PI/2));
        scenarioExperiments.put("56", new Location(points[7].x, points[7].y, "location", PI/2));
        scenarioExperiments.put("57", new Location(points[8].x, points[8].y, "location", PI/2));
        scenarioExperiments.put("58", new Location(points[9].x, points[9].y, "location", PI/2));
        scenarioExperiments.put("59", new Location(points[10].x, points[10].y, "location", PI/2));
        scenarioExperiments.put("60", new Location(points[11].x, points[11].y, "location", PI/2));
        scenarioExperiments.put("61", new Location(points[12].x, points[12].y, "location", PI/2));
        scenarioExperiments.put("62", new Location(points[13].x, points[13].y, "location", PI/2));
        scenarioExperiments.put("63", new Location(points[14].x, points[14].y, "location", PI/2));
        scenarioExperiments.put("64", new Location(points[6].x, points[6].y, "location", 1.5*PI));
        scenarioExperiments.put("65", new Location(points[7].x, points[7].y, "location", 1.5*PI));
        scenarioExperiments.put("66", new Location(points[8].x, points[8].y, "location", 1.5*PI));
        scenarioExperiments.put("67", new Location(points[9].x, points[9].y, "location", 1.5*PI));
        scenarioExperiments.put("68", new Location(points[10].x, points[10].y, "location", 1.5*PI));
        scenarioExperiments.put("69", new Location(points[11].x, points[11].y, "location", 1.5*PI));
        scenarioExperiments.put("70", new Location(points[12].x, points[12].y, "location", 1.5*PI));
        scenarioExperiments.put("71", new Location(points[13].x, points[13].y, "location", 1.5*PI));
        scenarioExperiments.put("72", new Location(points[14].x, points[14].y, "location", 1.5*PI));
        } catch (Exception e) {e.printStackTrace();}
      }
      
      scenarioExperiment = scenarioExperiments.get(experiment);
      if (scenarioExperiment == null) {
        exit();
      }
      
    }
    
  }
  
}

void draw() {
  
  background(152,190,100);
  
  try {
    
    for (int i=0; i < txBeamsInfluence.length; i++) {
      txBeamsInfluence[i].display();
    }
    
    tx.display();
    rx.display();
    radar1.display();
    radar2.display();
    
    for (int i=0; i < points.length; i++) {
      points[i].display();
    }
    
  } catch (Exception e) {
    e.printStackTrace();
    exception = true;
  }
  
  if (exception) {
    exit();
  }
  else {
  
    float currT, currAngle;
    PVector currPos;
    
    if (int(scenario) == 1) {
      Trajectory castScenarioExperiment = (Trajectory) scenarioExperiment;
      currT = castScenarioExperiment.evaluateT(msEllapsed);
      currAngle = castScenarioExperiment.evaluateTangentAngle(currT);
      currPos = castScenarioExperiment.evaluateBezier(currT);
    }
    else if (int(scenario) == 2 || int(scenario) == 3) {
      Location castScenarioExperiment = (Location) scenarioExperiment;
      currT = castScenarioExperiment.evaluateT(msEllapsed);
      currAngle = castScenarioExperiment.evaluateAngle(currT);
      currPos = castScenarioExperiment.evaluateLocation(currT);
    }
    else {
      currT = 0.0;
      currAngle = 0.0;
      currPos = new PVector(0,0);
    }
    
    if (int(scenario) == 1 || int(scenario) == 2 || int(scenario) == 3) {
  
      PVector[] boxModelcurrCoords = calculateTopleftBottomright(int(currPos.x), int(currPos.y), boxWidth, boxHeight);
      humanBoxModel.setTopleftBottomRight(boxModelcurrCoords[0], boxModelcurrCoords[1]);
      humanBoxModel.setAngle(currAngle);
      PVector[] boxModelcurrCorners = humanBoxModel.calculateTransformedCorners();
      humanBoxModel.display();
  
      if (msEllapsed == 10000) {
        success = true;
        exit();
      }
      else {
        
        for (int i = 0; i < txBeamsInfluence.length; i++) {
          
          //Deep copy corner vectors, Java passes PVector objects by reference to method calls
          PVector[] cornerCopy = new PVector[boxModelcurrCorners.length];

          for (int j = 0; j < boxModelcurrCorners.length; j++) {
              PVector corner = boxModelcurrCorners[j];
              cornerCopy[j] = new PVector(corner.x, corner.y, corner.z);
          }
          
          labelMatrix[msEllapsed/5][i] = int(txBeamsInfluence[i].isRectangleOverlapEllipse(cornerCopy[0], cornerCopy[1], cornerCopy[2], cornerCopy[3]));
        }
        
        msEllapsed += 5;
        
      }
    
    }
    else {
      exit();
    }

  }
    
}

void exit() {

  if (success && !exception) {
    
    try {
      println("Attempting To Save Array Contents To File...");
      
      BufferedWriter writer = new BufferedWriter(new FileWriter(String.format(outPath + "radar1_chirp profile%s_10s_Scen%s_activity#%s.txt", scenario, scenario, experiment), false));
      for(int i = 0; i < labelMatrix.length; i++) {
        StringBuilder sb = new StringBuilder();
        for (int j = 0; j < labelMatrix[i].length; j++) {
          sb.append(labelMatrix[i][j]).append(", ");
        }
        sb.setLength(sb.length() - 2); // Remove last comma append
        writer.write(sb.toString());
        writer.newLine();  // More Platform-independent that using write("\n");
      }
      writer.flush();
      writer.close();
      
      println("Saved Array To File Successfully...");
    } catch (IOException e) {
      println("Couldnt Save Array To File... ");
      e.printStackTrace();
    }

  }
  else if (!success && !exception) {
    println("Sketch requires three command line parameters: scenario (-sc <String>), experiment (-exp <String>), and data output path (-out <String>)."
            + " Allowed scenario values: 1, 2, 3. Allowed experiment values: 1-32 (scenario 1), 1-108 (scenario 2), 1-72 (scenario 3).");
  }
  
  super.exit();
}
