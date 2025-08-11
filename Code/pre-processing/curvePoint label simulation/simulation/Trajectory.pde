class Trajectory {
  
  ControlPoint[] controlPoints;
  String trajectoryType;
  int bpm, h, d;
  
  Trajectory(String type, String speed, int subjHeight, ControlPoint[] points) throws Exception {
    
    trajectoryType = type;
    h = subjHeight;
    d = int(effectiveDistance(points));
    
    if (trajectoryType.equals("straight")) {
      controlPoints = generateIntermediatePoints(points[0], points[1], 20);
    }
    else if (trajectoryType.equals("uturn")) {
      controlPoints = generateIntermediatePointsWithUTurn(points[0], points[1], 20);
    }
    else {
      throw new Exception("Unknown type variable. Allowed values: straight, uturn");
    }
    
    if (speed.equals("slow")) {
      bpm = 90;
    }
    else if (speed.equals("fast")) {
      bpm = 170;
    }
    else {
      throw new Exception("Unknown speed variable. Allowed values: slow, fast");
    }

  }
  
  void display() {
    
    noFill();
    stroke(255, 102, 0);
    beginShape();
    vertex(controlPoints[0].x, controlPoints[0].y);
    for (int i = 1; i < controlPoints.length; i += 3) {
      bezierVertex(controlPoints[i].x, controlPoints[i].y, controlPoints[i+1].x, controlPoints[i+1].y, controlPoints[i+2].x, controlPoints[i+2].y);
    }
    endShape();
    
  }
  
  PVector evaluateBezier(float t) {
    
    int segmentCount = (controlPoints.length - 1) / 3;
    int segment = min((int)(t * segmentCount), segmentCount - 1);
    t = (t - (float)segment / segmentCount) * segmentCount;
    
    ControlPoint p0 = controlPoints[segment * 3];
    ControlPoint p1 = controlPoints[segment * 3 + 1];
    ControlPoint p2 = controlPoints[segment * 3 + 2];
    ControlPoint p3 = controlPoints[segment * 3 + 3];
    
    float x = bezierPoint(p0.x, p1.x, p2.x, p3.x, t);
    float y = bezierPoint(p0.y, p1.y, p2.y, p3.y, t);
    
    return new PVector(x, y);
    
  }
  
  float evaluateTangentAngle(float t) {
     
    int segmentCount = (controlPoints.length - 1) / 3;
    int segment = min((int)(t * segmentCount), segmentCount - 1);
    t = (t - (float)segment / segmentCount) * segmentCount;
    
    ControlPoint p0 = controlPoints[segment * 3];
    ControlPoint p1 = controlPoints[segment * 3 + 1];
    ControlPoint p2 = controlPoints[segment * 3 + 2];
    ControlPoint p3 = controlPoints[segment * 3 + 3];
    
    float tx = bezierTangent(p0.x, p1.x, p2.x, p3.x, t);
    float ty = bezierTangent(p0.y, p1.y, p2.y, p3.y, t);
    
    float a = atan2(ty, tx);
    a += PI;
    return a;
    
  }
  
  // height h (cm), ellapsed time t (ms), travellable distance d (cm)
  // Formula based on Bing Copilot prompt: 
  float evaluateT(int t) {
  
    float bps = bpm / 60.0;
    float ts = t / 1000.0;
    float step = stepLenWalking();
    float dt = min(bps * step * ts, float(d));
    return dt / float(d);
  
  }
  
  // Assumed gender male, height h (cm): https://digitalcommons.wku.edu/ijes/vol3/iss1/2/, https://www.verywellfit.com/set-pedometer-better-accuracy-3432895
  // Stride is assumed to be two steps (see Verywellfit text)
  float stepLenWalking() {
  
    float h_in = h / 2.54; // cm to inch conversion
    float step_in = (h_in * 0.415) / 2; // account for step (half stride) being synced on metronome beat
    return step_in * 2.54;
  
  }
  
  float effectiveDistance(ControlPoint[] p) {
  
    float totalDistance = 0.0;
    for (int i = 0; i < p.length - 1; i++) {
      totalDistance += PVector.dist(new PVector(p[i].x, p[i].y), new PVector(p[i+1].x, p[i+1].y));
    }
  
    if (trajectoryType.equals("uturn")) {
      // If the scenario type is "U-turn", we double the total distance
      // as the path is traversed twice (there and back)
      totalDistance *= 2;
    }
    
    return totalDistance;

  }
  
  ControlPoint[] generateIntermediatePointsWithUTurn(ControlPoint start, ControlPoint end, int numIntermediatePoints) {
    
    ControlPoint[] points = new ControlPoint[(numIntermediatePoints + 2) * 2 - 1];
    ControlPoint[] firstHalf = generateIntermediatePoints(start, end, numIntermediatePoints);
    ControlPoint[] secondHalf = generateIntermediatePoints(end, start, numIntermediatePoints);
    arrayCopy(firstHalf, 0, points, 0, firstHalf.length);
    arrayCopy(secondHalf, 1, points, firstHalf.length, secondHalf.length - 1);  // Skip the first point of the second half to avoid duplication
    return points;

  }
  
  ControlPoint[] generateIntermediatePoints(ControlPoint start, ControlPoint end, int numIntermediatePoints) {
    
    ControlPoint[] points = new ControlPoint[numIntermediatePoints + 2];
    points[0] = start;
    points[points.length - 1] = end;
    
    for (int i = 1; i <= numIntermediatePoints; i++) {
      float t = i / (float)(numIntermediatePoints + 1);
      float x = lerp(start.x, end.x, t);
      float y = lerp(start.y, end.y, t);
      points[i] = new ControlPoint(int(x), int(y), "control");
    }
    
    return points;
    
  }
  
}
