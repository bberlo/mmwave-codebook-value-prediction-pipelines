class FresnelEllipsoid {

  PVector txPos, rxPos;
  float vRadius;
  
  FresnelEllipsoid(ControlPoint tx, ControlPoint rx, float r) {

    txPos = new PVector(tx.x, tx.y);
    rxPos = new PVector(rx.x, rx.y);
    vRadius = r;

  }

  void display() {
    
    pushMatrix();
    fill(216, 27, 212);
    stroke(0);
    translate((txPos.x + rxPos.x)/2, (txPos.y + rxPos.y)/2);
    float hRadius = PVector.dist(txPos, rxPos)/2;
    rotate(calculateAngle(txPos, rxPos));
    ellipse(0, 0, hRadius*2, vRadius*2);
    popMatrix();

  }

  // https://www.geometrictools.com/Documentation/IntersectionRectangleEllipse.pdf
  // Alternative: matrix projection to cilinder/triangle scenario https://gamedev.stackexchange.com/questions/12836/ellipsoid-v-box-collision-detection
  boolean isRectangleOverlapEllipse(PVector topLeft, PVector topRight, PVector bottomRight, PVector bottomLeft) {
    
    // Check if any of the corners of the rectangle are inside the ellipse
    if (isCoorInsideEllipse(topLeft.x, topLeft.y) || isCoorInsideEllipse(topRight.x, topRight.y) ||
        isCoorInsideEllipse(bottomLeft.x, bottomLeft.y) || isCoorInsideEllipse(bottomRight.x, bottomRight.y)) {
      return true;
    }

    // Check if any of the sides of the rectangle intersect the ellipse
    PVector[] rectangleCorners = {topLeft, topRight, bottomRight, bottomLeft};
    for (int i = 0; i < rectangleCorners.length; i++) {
      PVector corner1 = rectangleCorners[i];
      PVector corner2 = rectangleCorners[(i+1)%rectangleCorners.length]; // Next corner in the rectangle

      // Calculate the distance from the center of the ellipse to the line formed by the side of the rectangle
      float centerX = (txPos.x + rxPos.x)/2;
      float centerY = (txPos.y + rxPos.y)/2;
      PVector center = new PVector(centerX, centerY);
      float angle = calculateAngle(txPos, rxPos);

      // Deep copy to prevent unwanted PVector object pass by reference value alterations
      PVector corner1Copy = corner1.copy();
      PVector corner2Copy = corner2.copy();

      // Translate the point to the origin, rotate it, then translate it back
      corner1Copy.sub(centerX, centerY);
      corner1Copy.rotate(-angle);
      corner1Copy.add(centerX, centerY);

      corner2Copy.sub(centerX, centerY);
      corner2Copy.rotate(-angle);
      corner2Copy.add(centerX, centerY);

      // Calculate the distance from the center to the line formed by the side of the rectangle
      float distance = PVector.dist(PVector.lerp(corner1Copy, corner2Copy, PVector.dot(PVector.sub(center, corner1Copy), PVector.sub(corner2Copy, corner1Copy)) / PVector.dist(corner1Copy, corner2Copy)), center);

      // Check if the distance is less than the radius of the ellipse
      if (distance < vRadius) {
        return true;
      }
    }
    
    // Check if the rectangle center is inside the ellipse
    float rectCenterX = (topLeft.x + bottomRight.x) / 2;
    float rectCenterY = (topLeft.y + bottomRight.y) / 2;
    return isCoorInsideEllipse(rectCenterX, rectCenterY);

  }

  boolean isCoorInsideEllipse(float x, float y) {

    PVector point = new PVector(x, y);
    float hRadius = PVector.dist(txPos, rxPos)/2;
    float centerX = (txPos.x + rxPos.x)/2;
    float centerY = (txPos.y + rxPos.y)/2;
    float angle = calculateAngle(txPos, rxPos);
    
    // Deep copy to prevent unwanted PVector object pass by reference value alterations
    PVector pointCopy = point.copy();

    // Translate the point to the origin, rotate it, then translate it back
    pointCopy.sub(centerX, centerY);
    pointCopy.rotate(-angle);
    pointCopy.add(centerX, centerY);

    return sq((pointCopy.x - centerX)/hRadius) + sq((pointCopy.y - centerY)/vRadius) <= 1;

  }

  float calculateAngle(PVector p1, PVector p2) {
    
    return atan2(p2.y - p1.y, p2.x - p1.x);

  }
  
}
