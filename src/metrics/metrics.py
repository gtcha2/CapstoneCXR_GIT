
import evaluate

##TODO: add in radgraph metrics.... somehow download it. 
class metricScorer:
    
    def __init__(self,**kwargs):
        #unpack the metric dictionary
        self.localMetricKeys=kwargs.keys()
        self.__dict__.update(kwargs)
        self.loadMetrics()
    def loadMetrics(self):
        for x in self.localMetricKeys:
            # should load in the metric of choidce
            try:
                setattr(self,x, evaluate.load(x))
            except:
                print(f"{x} is not a loadable metric with eval")
                continue
    def scoreMetrics(self,**values):
          #predictions and refernces

      for x in self.localMetricKeys:
        
        metric=getattr(self,x)
        if x == "bertscore":
          print(metric.compute(predictions=values["predictions"],references=values["references"],lang="en"))
          continue
        if not metric is None:
          print(metric.compute(predictions=values["predictions"],references=values["references"]))
    

    
        
        
    