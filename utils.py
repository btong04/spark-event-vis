from collections import namedtuple
import json
import pandas as pd
from functools import reduce
import pyspark
spark = pyspark.sql.SparkSession.builder.getOrCreate()

MetricMeta = namedtuple('MetricMeta', 'name kind unit')
metric_metas = [
    MetricMeta('GPU decode time', 'time', 'ns'),
    MetricMeta('GPU time', 'time', 'ns'),
    MetricMeta('avg hash probe bucket list iters', 'count', 'iterations'),
    MetricMeta('buffer time', 'time', 'ns'),
    MetricMeta('build side size', 'size', 'bytes'),
    MetricMeta('build time', 'time', 'ms'),
    MetricMeta('collect batch time', 'time', 'ns'),
    MetricMeta('concat batch time', 'time', 'ns'),
    MetricMeta('data size', 'size', 'bytes'),
    MetricMeta('duration', 'time', 'ms'),
    MetricMeta('fetch wait time', 'time', 'ms'),
    MetricMeta('internal.metrics.diskBytesSpilled', 'size', 'bytes'),
    MetricMeta('internal.metrics.executorCpuTime', 'time', 'ns'),
    MetricMeta('internal.metrics.executorDeserializeCpuTime', 'time', 'ns'),
    MetricMeta('internal.metrics.executorDeserializeTime', 'time', 'ms'),
    MetricMeta('internal.metrics.executorRunTime', 'time', 'ms'),
    MetricMeta('internal.metrics.input.recordsRead', 'count', 'records'),
    MetricMeta('internal.metrics.jvmGCTime', 'time', 'ms'),
    MetricMeta('internal.metrics.memoryBytesSpilled', 'size', 'bytes'),
    MetricMeta('internal.metrics.output.bytesWritten', 'size', 'bytes'),
    MetricMeta('internal.metrics.output.recordsWritten', 'count', 'records'),
    MetricMeta('internal.metrics.peakExecutionMemory', 'size', 'bytes'),
    MetricMeta('internal.metrics.resultSerializationTime', 'time', 'ms'),
    MetricMeta('internal.metrics.resultSize', 'size', 'bytes'),
    MetricMeta('internal.metrics.shuffle.read.fetchWaitTime', 'time', 'ms'),
    MetricMeta('internal.metrics.shuffle.read.localBlocksFetched', 'count', 'blocks'),
    MetricMeta('internal.metrics.shuffle.read.localBytesRead', 'size', 'bytes'),
    MetricMeta('internal.metrics.shuffle.read.recordsRead', 'count', 'records'),
    MetricMeta('internal.metrics.shuffle.read.remoteBlocksFetched', 'count', 'blocks'),
    MetricMeta('internal.metrics.shuffle.read.remoteBytesRead', 'size', 'bytes'),
    MetricMeta('internal.metrics.shuffle.read.remoteBytesReadToDisk', 'size', 'bytes'),
    MetricMeta('internal.metrics.shuffle.write.bytesWritten', 'size', 'bytes'),
    MetricMeta('internal.metrics.shuffle.write.recordsWritten', 'count', 'records'),
    MetricMeta('internal.metrics.shuffle.write.writeTime', 'time', 'ns'),
    MetricMeta('join output rows', 'count', 'rows'),
    MetricMeta('join time', 'time', 'ns'),
    MetricMeta('local blocks read', 'count', 'blocks'),
    MetricMeta('local bytes read', 'size', 'bytes'),
    MetricMeta('number of input columnar batches', 'count', 'batches'),
    MetricMeta('number of input rows', 'count', 'rows'),
    MetricMeta('number of output columnar batches', 'count', 'batches'),
    MetricMeta('number of output rows', 'count', 'rows'),
    MetricMeta('peak device memory', 'size', 'bytes'),
    MetricMeta('peak memory', 'size', 'bytes'),
    MetricMeta('records read', 'count', 'records'),
    MetricMeta('remote blocks read', 'count', 'blocks'),
    MetricMeta('remote bytes read', 'size', 'bytes'),
    MetricMeta('scan time', 'time', 'ms'),
    MetricMeta('shuffle bytes written', 'size', 'bytes'),
    MetricMeta('shuffle records written', 'count', 'records'),
    MetricMeta('shuffle write time', 'time', 'ns'),
    MetricMeta('spill size', 'size', 'bytes'),
    MetricMeta('time in aggregation build', 'time', 'ms'),
    MetricMeta('time in batch concat', 'time', 'ns'),
    MetricMeta('time in compute agg', 'time', 'ns'),
    MetricMeta('total time', 'time', 'ns'),
    MetricMeta('write time', 'time', 'ns'),
    MetricMeta('internal.metrics.input.bytesRead', 'size', 'bytes'),
    MetricMeta('sort time', 'time', 'ns')
]

# Create dictionary for units:
attr_units = pd.DataFrame(metric_metas)
attr_units_dict = {}
attr_units_kind = attr_units.groupby('kind')
for gid,grp in attr_units_kind:
    tmp = {}
    for unit_gid, unit_grp in grp.groupby('unit'):
        tmp[unit_gid] = list(unit_grp['name'].values)
        attr_units_dict[gid] = tmp

time_cols = attr_units[attr_units['kind']=='time']['name'].values
size_cols = attr_units[attr_units['kind']=='size']['name'].values
count_cols = attr_units[attr_units['kind']=='count']['name'].values

def convert_time_to_seconds(df):
    """
    Convert time to seconds.
    """

    # Time unit conversion to seconds:
    for unit in attr_units_dict['time'].keys():
        sel_cols = set(df.columns).intersection(attr_units_dict['time'][unit])

        if unit=='ms':
            df.loc[:,sel_cols] = df.loc[:,sel_cols]/1E3
        elif unit=='ns':
            df.loc[:,sel_cols] = df.loc[:,sel_cols]/1E9
        elif unit=='s':
            print('Units assumed to be in seconds.')
            # Do nothing
        else:
            raise ValueError('Time unit not recognized. Choose ms or ns.')
    return(df)


def etl_stage_metrics(filename, label, verbose=False):
    """
    Read and parse spark history server logs directly using spark SQL. Make sure logs only include workload of interest. Entire contents of file will be aggregated.
    
    filename: str
        Full path to file. Assumes JSON file format.

    label: str
        Label for identifying workload for plotting purposes.

    Event log parsing based on:
    https://github.com/LucaCanali/Miscellaneous/blob/master/Spark_Notes/Spark_EventLog.md

    Description of taskMetrics can be found here: 
    https://spark.apache.org/docs/latest/monitoring.html#executor-task-metrics
    https://github.com/LucaCanali/Miscellaneous/blob/master/Spark_Notes/Spark_TaskMetrics.md
    """
    
    # Load data:
    df = spark.read.json(filename)

    if verbose==True:
        # Look at types of events:
        df.select("Event").groupBy("Event").count().show(30,False)

    # Use spark SQL to flatten attributes:
    df2 = df.filter("Event='SparkListenerStageCompleted'").select("`Stage Info`.*")
    df2.createOrReplaceTempView("t2")

    df4 = spark.sql("select `Submission Time`, `Completion Time`, `Number of Tasks`, `Stage ID`, `Stage Name` , t3.col.* from t2 lateral view explode(Accumulables) t3")
    df4.createOrReplaceTempView("t4")

    # Aggregate stage info metrics values
    summary_report = spark.sql("select Name, sum(Value) as value from t4 group by Name order by Name").toPandas()
    summary_report = summary_report.set_index('Name').T

    # Output results:
    output = {
        't2': df2.toPandas(),
        't4': df4.toPandas(),
        'summary': summary_report
    }
    return(output)