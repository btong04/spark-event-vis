import os
import pandas as pd
import json
import sparkmeasure as spm
import pyspark

# Create SparkContext and submit custom jars:
cwd = os.getcwd()
sparkMeasure_jar = os.path.join(cwd, 'spark-measure_2.12-0.18-SNAPSHOT-mod.jar')

# Create SparkContext (i.e., spark local cluster by default if spark.master not set):
config = pyspark.SparkConf().setAll([
    ('spark.driver.memory','4G'), 
    ('spark.executor.memory','2G'), 
    ('spark.driver.extraClassPath',sparkMeasure_jar), 
    ('spark.executor.extraClassPath',sparkMeasure_jar)]
    )
sc = pyspark.SparkContext(master='local[*]', conf=config)

# Use sc.stop() to kill SparkContext.

# Create SparkSession from existing SparkContext:
spark = pyspark.sql.SparkSession(sc)

# Initialize sparkMeasure metrics:
stagemetrics = spm.StageMetrics(spark)
taskmetrics = spm.TaskMetrics(spark)

# Define cell and line magic to wrap the instrumentation
from IPython.core.magic import (register_line_magic, register_cell_magic, register_line_cell_magic)

@register_line_cell_magic
def stageMeasure(line, cell=None):
    "run and measure spark workload. Use: %stageMeasure or %%stageMeasure"
    val = cell if cell is not None else line
    stagemetrics.begin()
    eval(val)
    stagemetrics.end()
#     stagemetrics.print_report()
    
@register_line_cell_magic
def taskMeasure(line, cell=None):
    "run and measure spark workload. Use: %taskMeasure or %%taskMeasure"
    val = cell if cell is not None else line
    taskmetrics.begin()
    eval(val)
    taskmetrics.end()
#     taskmetrics.print_report()
    

def save_stagemetrics(output_dir):
    # Create PerfStageMetrics spark dataframe. 
    perf_spdf = stagemetrics.create_stagemetrics_DF("PerfStageMetrics")

    # Save sparkMeasure "standard" output:
    stagemetrics.save_data(perf_spdf.orderBy("jobId", "stageId"), output_dir)
    
def save_taskmetrics(output_dir):
    # Create PerfStageMetrics spark dataframe. 
    perf_spdf = taskmetrics.create_taskmetrics_DF("PerfTaskMetrics")

    # Save sparkMeasure "standard" output:
    taskmetrics.save_data(perf_spdf.orderBy("jobId", "stageId", "index"), output_dir, "json")


def save_custom_stagemetrics(output_filename):
    """
    Save output of PerfStageMetrics to custom JSON file.
    Note that stageMeasure() needs to be ran to generate the PerfStageMetrics table before hand.
    """
    # Use spark to read metrics temp table and convert to pandas/json.
    # PerfStageMetrics is temp table created which contains summary of stage metrics.
    perf_stagemetrics = spark.sql("select * from PerfStageMetrics").toPandas()


    # Summarize stage metrics for entire application. Results same as stagemetrics.print_report().
    aggregatedDF = stagemetrics.aggregate_stagemetrics_DF("PerfStageMetrics")

    # aggregatedDF cannot be converted to pandas dataframe: 'SparkSession' object has no attribute '_conf'
    # Work around using other operations. Same data as found in stagemetrics.print_report().
    perf_summary = dict(zip(aggregatedDF.columns, aggregatedDF.first()))
    
    # Construct nested dictionary as output. Convert results to json afterwards.
    perf_out = {}

    perf_out['stageSummary'] = perf_summary
    perf_out['stageMetrics'] = perf_stagemetrics.to_dict()

    with open(output_filename, 'w') as json_file:
        json.dump(perf_out, json_file)
    return(print(output_filename.split('/')[-1] + ' saved.'))

def save_custom_taskmetrics(output_filename):
    """
    Save output of PerfTaskMetrics to custom JSON file.
    Note that taskMeasure() needs to be ran to generate the PerfTaskMetrics table before hand.
    """

    # Use spark to read metrics temp table and convert to pandas/json.
    # PerfTaskMetrics is temp table created which contains summary of stage metrics.
    perf_taskmetrics = spark.sql("select * from PerfTaskMetrics").toPandas()


    # Summarize stage metrics for entire application. Results same as stagemetrics.print_report().
    aggregatedDF = taskmetrics.aggregate_taskmetrics_DF("PerfTaskMetrics")

    # aggregatedDF cannot be converted to pandas dataframe: 'SparkSession' object has no attribute '_conf'
    # Work around using other operations. Same data as found in taskmetrics.print_report().
    perf_summary = dict(zip(aggregatedDF.columns, aggregatedDF.first()))
    
    # Construct nested dictionary as output. Convert results to json afterwards.
    perf_out = {}

    perf_out['taskSummary'] = perf_summary
    perf_out['taskMetrics'] = perf_taskmetrics.to_dict()

    with open(output_filename, 'w') as json_file:
        json.dump(perf_out, json_file)
    return(print(output_filename.split('/')[-1] + ' saved.'))


######################################
#### Flight Recorder JSON Methods ####
######################################
def load_flightRecorder_json(scope, filepath, verbose=False):
    """
    Load sparkMetrics flight recorder JSON's from disk. Register dataframe as TempView for either PerfTaskMetrics or PerfStageMetrics. 
    
    scope: str  
        Select 'task' or 'stage' for metric scope.
        
    filepath: str  
        Fully qualified file path for JSON file.
        
    verbose: bool  
        Prints table schema.
    """
    
    load_perf_metrics = spark.read.option('inferSchema', True).json(filepath, multiLine=True)
    load_perf_metrics.createOrReplaceTempView('Perf'+scope.title()+'Metrics')
    if verbose:
        load_perf_metrics.printSchema()
    return(load_perf_metrics.toPandas())
    
def agg_flightRecorder_json(scope, beginSnapshot=0, endSnapshot=0):
    """
    Aggregates metrics from PerfTaskMetrics or PerfStageMetrics from temp view table in spark. 
    Modified from taskmetrics.scala, aggregateTaskMetrics(). Can't use python bindings provided in spm.aggregate_taskmetrics_DF() to load from disk.
    
    Expect JSON read into spark and registered as df.createOrReplaceTempView("PerfTaskMetrics").
    
    scope: str  
        Select 'task' or 'stage' for metric scope.
        
    beginSnapshot: int64  
        Start of time window in epoch milliseconds.
        
    endSnapshot: int64  
        End of time window in epoch milliseconds. 
    
    TODO: pass beginSnapshot and endSnapshot directly so that spm.taskmetrics.aggregate_taskmetrics_DF() can be ran directly on:
    "where launchTime >= $beginSnapshot and finishTime <= $endSnapshot")
    """
    
    if (beginSnapshot == 0) & (endSnapshot == 0):
        # Include everything.
        time_filter = ""
    else:
        if scope == 'task':
            time_filter = "where launchTime >= "+str(beginSnapshot)+" and finishTime <= "+str(endSnapshot)
        elif scope == 'stage':
            time_filter = "where submissionTime >= "+str(beginSnapshot)+" and completionTime <= "+str(endSnapshot)
        
    if scope == 'task':
        results = spark.sql("select count(*) as numtasks, " +
              "max(finishTime) - min(launchTime) as elapsedTime, sum(duration) as duration, sum(schedulerDelay) as schedulerDelayTime, " +
              "sum(executorRunTime) as executorRunTime, sum(executorCpuTime) as executorCpuTime, " +
              "sum(executorDeserializeTime) as executorDeserializeTime, sum(executorDeserializeCpuTime) as executorDeserializeCpuTime, " +
              "sum(resultSerializationTime) as resultSerializationTime, sum(jvmGCTime) as jvmGCTime, "+
              "sum(shuffleFetchWaitTime) as shuffleFetchWaitTime, sum(shuffleWriteTime) as shuffleWriteTime, " +
              "sum(gettingResultTime) as gettingResultTime, " +
              "max(resultSize) as resultSize, " +
              "sum(diskBytesSpilled) as diskBytesSpilled, sum(memoryBytesSpilled) as memoryBytesSpilled, " +
              "max(peakExecutionMemory) as peakExecutionMemory, sum(recordsRead) as recordsRead, sum(bytesRead) as bytesRead, " +
              "sum(recordsWritten) as recordsWritten, sum(bytesWritten) as bytesWritten, " +
              "sum(shuffleRecordsRead) as shuffleRecordsRead, sum(shuffleTotalBlocksFetched) as shuffleTotalBlocksFetched, "+
              "sum(shuffleLocalBlocksFetched) as shuffleLocalBlocksFetched, sum(shuffleRemoteBlocksFetched) as shuffleRemoteBlocksFetched, "+
              "sum(shuffleTotalBytesRead) as shuffleTotalBytesRead, sum(shuffleLocalBytesRead) as shuffleLocalBytesRead, " +
              "sum(shuffleRemoteBytesRead) as shuffleRemoteBytesRead, sum(shuffleRemoteBytesReadToDisk) as shuffleRemoteBytesReadToDisk, " +
              "sum(shuffleBytesWritten) as shuffleBytesWritten, sum(shuffleRecordsWritten) as shuffleRecordsWritten " +
              "from PerfTaskMetrics " + time_filter)
        
    elif scope == 'stage':
        results = spark.sql("select count(*) as numStages, sum(numTasks) as numTasks, " +
              "max(completionTime) - min(submissionTime) as elapsedTime, sum(stageDuration) as stageDuration , " +
              "sum(executorRunTime) as executorRunTime, sum(executorCpuTime) as executorCpuTime, " +
              "sum(executorDeserializeTime) as executorDeserializeTime, sum(executorDeserializeCpuTime) as executorDeserializeCpuTime, " +
              "sum(resultSerializationTime) as resultSerializationTime, sum(jvmGCTime) as jvmGCTime, "+
              "sum(shuffleFetchWaitTime) as shuffleFetchWaitTime, sum(shuffleWriteTime) as shuffleWriteTime, " +
              "max(resultSize) as resultSize, " +
              "sum(diskBytesSpilled) as diskBytesSpilled, sum(memoryBytesSpilled) as memoryBytesSpilled, " +
              "max(peakExecutionMemory) as peakExecutionMemory, sum(recordsRead) as recordsRead, sum(bytesRead) as bytesRead, " +
              "sum(recordsWritten) as recordsWritten, sum(bytesWritten) as bytesWritten, " +
              "sum(shuffleRecordsRead) as shuffleRecordsRead, sum(shuffleTotalBlocksFetched) as shuffleTotalBlocksFetched, "+
              "sum(shuffleLocalBlocksFetched) as shuffleLocalBlocksFetched, sum(shuffleRemoteBlocksFetched) as shuffleRemoteBlocksFetched, "+
              "sum(shuffleTotalBytesRead) as shuffleTotalBytesRead, sum(shuffleLocalBytesRead) as shuffleLocalBytesRead, " +
              "sum(shuffleRemoteBytesRead) as shuffleRemoteBytesRead, sum(shuffleRemoteBytesReadToDisk) as shuffleRemoteBytesReadToDisk, " +
              "sum(shuffleBytesWritten) as shuffleBytesWritten, sum(shuffleRecordsWritten) as shuffleRecordsWritten " +
              "from PerfStageMetrics " + time_filter)
    return(results.toPandas())

