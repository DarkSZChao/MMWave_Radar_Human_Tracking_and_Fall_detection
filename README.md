Abstract
This study explores an indoor system for tracking multiple humans and detecting falls, employing three Millimeter-Wave radars from Texas Instruments placed on x-y-z surfaces. Compared to wearables and camera methods, Millimeter-Wave radar is not plagued by mobility inconvenience, lighting conditions, or privacy issues. We establish a real-time framework to integrate signals received from these radars, allowing us to track the position and body status of human targets non-intrusively. To ensure the overall accuracy of our system, we conduct an initial evaluation of radar characteristics, covering aspects such as resolution, interference between radars, and coverage area. Additionally, we introduce innovative strategies, including Dynamic DBSCAN clustering based on signal energy levels, a probability matrix for enhanced target tracking, target status prediction for fall detection, and a feedback loop for noise reduction. We conducted an extensive evaluation using over 300 minutes of data, which equates to approximately 360,000 frames. Our prototype system exhibited remarkable performance, achieving a precision of 98.9\% for tracking a single target and 96.5\% and 94.0\% for tracking two and three targets in human tracking scenarios. Moreover, in the field of human fall detection, the system demonstrated a high accuracy of 98.2\%, underscoring its effectiveness in distinguishing falls from other statuses.