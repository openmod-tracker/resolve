# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---
# %% jupyter={"source_hidden": true}
import datetime as dt
import logging
import os
import sys
from base64 import b64encode
from datetime import datetime
from datetime import timedelta
from functools import cache
from pathlib import Path

import pandas as pd
import panel as pn
import panel.widgets as pnw
from hera.workflows import Parameter
from hera.workflows import Workflow
from hera.workflows import WorkflowsService
from hera.workflows.models import Arguments
from hera.workflows.models import WorkflowTemplateRef

SERVICEACCOUNT_TOKEN_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/token"  # nosec

NAMESPACE = os.environ.get('NAMESPACE', 'default')

ARGO_URL = f"http://argo.{os.environ['ENKAP_PLATFORM']}.{os.environ['ENKAP_INSTANCE']}.ethree.cloud"

ARGO_SERVER = 'http://argo-workflows-server:2746'
ARGO_TOKEN = Path(SERVICEACCOUNT_TOKEN_FILE).read_text()

from hera.shared import global_config
global_config.namespace = NAMESPACE
global_config.host = ARGO_SERVER
global_config.token = ARGO_TOKEN


pn.extension('terminal', design="material", sizing_mode="stretch_width")
log = logging.getLogger('panel.cloudrunner')

_STYLESHEET = """
:nth-child(2 of .bk-panel-models-layout-Column) {
  flex-grow: 0
}
"""


# %% editable=true slideshow={"slide_type": ""} jupyter={"source_hidden": true}
class ModelSettings:


    def __init__(self,
                 base: str = '..',
                 datadir: str = 'data',
                 model:str = os.environ.get('E3_KIT_MODEL')):
        self.model = model
        self.base = Path(base).resolve()
        self.datadir = datadir

    @property
    def settings_path(self):
        return self.base / self.datadir / 'settings' / self.model

    @property
    def base_path(self):
        return self.base

    @property
    def workflow_template_path(self):
        return self.base / 'workflows' / f"{self.model}.workflow.yaml"


# %%
model = ModelSettings()


# %% jupyter={"source_hidden": true}
def _load_cases(model):
    return sorted(map(lambda x: x.name, filter(Path.is_dir, model.settings_path.iterdir())))


@cache
def _load_workflow_template(model):
    return model.workflow_template_path.read_text()

def _submit_workflow(model, case, cpu, memory, ephemeral_storage):
    ts = datetime.now().strftime('%Y%m%d-%H%M-')

    # Create workflow based on a a workspace template
    wft = WorkflowTemplateRef(name=os.environ['E3_WORKSPACE'])
    wf = Workflow(
        generate_name=f"{os.environ['E3_OWNER']}-{os.environ['E3_KIT_MODEL']}-{ts}",
        workflow_template_ref=wft,
    )

    # Create workflow from YAML template on disk
    # wf = Workflow.from_yaml(_load_workflow_template(model))

    # Set workflow parameters
    datahash = b64encode((model.base_path / "data.dvc").read_text().encode("utf-8")).decode("ascii")
    wf.arguments = Arguments(
        parameters=[
            Parameter(name="datahash", value=datahash),
            Parameter(name="owner", value=os.environ["E3_OWNER"]),
            Parameter(name="case", value=case),
            Parameter(name="cpu", value=cpu),
            Parameter(name="memory", value=memory),
            Parameter(name="ephemeralStorage", value=ephemeral_storage),
            Parameter(name="args", value="--log-json --solver-name=gurobi"),
        ]
    )
    # next(filter(lambda x: x.name == 'owner', wf.arguments.parameters)).value =
    # next(filter(lambda x: x.name == 'case', wf.arguments.parameters)).value = case

    # Submit workfolw
    return wf.create();


def map_workflow(w):
    try:
        return [
            w.status.started_at.__root__,
            w.metadata.name,
            w.metadata.labels['beta.ethree.cloud/project'],
            w.metadata.labels['beta.kit.ethree.cloud/model'].upper(),
            w.metadata.labels['beta.ethree.cloud/owner'],
            w.metadata.labels['beta.kit.ethree.cloud/case'],
            w.status.nodes[w.metadata.name].phase if w.status.nodes else w.status.phase
        ]
    except Exception:
        log.error(f"Error while extracting workflow information: {w.metadata.name}: {sys.exception()}")
        return []

class CloudRunner():
    def __init__(self, model):
        self._ws = WorkflowsService()

        progress = pn.indicators.Progress(value = 0, bar_color = 'primary', active = False)

        def uneditable(active):
            return active

        def unsubmittable(cases, active):
            return len(cases) == 0 or active

        selector = pnw.CrossSelector(
            name='Case Selector',
            options=_load_cases(model),
            disabled=pn.bind(uneditable, progress.param.active),
            stylesheets=[_STYLESHEET])
        submit = pnw.Button(
            name='Submit',
            icon='send',
            button_type='primary',
            disabled=pn.bind(unsubmittable, selector, progress.param.active)
        )

        def on_submit(event):
            progress.active = True
            progress.value = -1

            error = False
            for case in selector.value:
                log.debug(f"Submitting case: {case}...")
                try:
                    wf = _submit_workflow(model, case,
                                         self._cpu.value, self._memory.value, self._ephemeral_storage.value)
                    log.info(f"Succesfully submitted case: {case}: {wf.metadata.name}")
                except:
                    error = True
                    log.error(f"Exception while submitting case: {case}: {sys.exception()}")

            progress.value = 0
            progress.active = False
            if not error:
                selector.value = []

        submit.on_click(on_submit)

        params = self._get_workflow_parameters()
        self._cpu = pn.widgets.RadioButtonGroup(value=params['cpu']['value'], options=params['cpu']['values'])
        self._memory = pn.widgets.RadioButtonGroup(value=params['memory']['value'], options=params['memory']['values'])
        self._ephemeral_storage = pn.widgets.RadioButtonGroup(value=params['ephemeralStorage']['value'], options=params['ephemeralStorage']['values'])

        self._display = pn.Column(
            pn.Row(pn.pane.Markdown('### CPU'), self._cpu),
            pn.Row(pn.pane.Markdown('### Memory'), self._memory),
            pn.Row(pn.pane.Markdown('### Ephemeral Storage'), self._ephemeral_storage),
            selector, progress, submit
        )

    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._display._repr_mimebundle_(include, exclude)



    def _get_workflow_parameters(self):
        wftmpl = self._ws.get_workflow_template(name=os.environ['E3_WORKSPACE'])
        params = filter(lambda x: x.enum is not None and x.value is not None, wftmpl.spec.arguments.parameters)

        return {v.name: {'values': v.enum, 'value': v.value} for v in params}

# %%

CloudRunner(model)


# %% jupyter={"source_hidden": true}
class CloudWatcher():
    def __init__(self):
        self._ws = WorkflowsService()
        self._data = None

        self._range = pnw.RangeSlider(name = 'Time Range (hours)', end = 240.0, step = 1.0)
        self._statuses = pnw.CheckButtonGroup(name='Statuses', options=['Pending', 'Running', 'Succeeded', 'Failed'])
        self._reload =  pnw.Button( name='Reload', icon='refresh', button_type='primary')

        def _update(range, statuses, reload):
            if (self._data is None or reload):
                self.reload()
            return self.workflows

        data = pn.bind(_update, range=self._range, statuses=self._statuses, reload=self._reload)
        content = pn.pane.DataFrame(data, escape = False)

        controls = pn.Row(self._range, self._statuses)
        self._display = pn.Column(controls, self._reload, content)


    def _repr_mimebundle_(self, include=None, exclude=None):
        return self._display._repr_mimebundle_(include, exclude)

    @property
    def workflows(self):
        if self._data is None:
            self.reload()
        data = self._data

        statuses = self._statuses.value
        if len(statuses) > 0:
            data = data[self._data["Status"].isin(statuses)]

        range = (datetime.now(dt.UTC) - timedelta(hours=self._range.value[0]),
                 datetime.now(dt.UTC) - timedelta(hours=self._range.value[1]))
        data = data[data["Started At"].apply(lambda x: x.to_pydatetime() < range[0] and x.to_pydatetime() > range[1])]

        return data if data.empty else data.sort_values(by=["Started At"], ascending=[False])

    def reload(self):
        result = map(map_workflow, self._ws.list_workflows().items or [])

        df = pd.DataFrame(result, columns=["Started At", "ID", "Project", "Model", "Owner", "Case", "Status"])
        df["Links"] = f"<a href='{ARGO_URL}/workflows/{NAMESPACE}/" + df['ID'] + "' target='_blank'>A</a>"
        df["Links"] += " <a href='https://app.datadoghq.com/dashboard/p48-8x8-2p5?&tpl_var_runId%5B0%5D=" + df['ID'] + "&live=true' target='_blank'>D</a>"
        df["Links"] += " <a href='https://app.datadoghq.com/logs?query=kube_ownerref_name%3A" + df['ID'] + "&live=true' target='_blank'>L</a>"

        self._data = df.set_index("ID")

# %%
CloudWatcher()

# %% jupyter={"source_hidden": true}
pn.widgets.Debugger(name='RUNNER Console', level=logging.INFO)
