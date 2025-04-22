# -*- coding: utf-8 -*-

__all__ = [
    "ExplainerTabsLayout",
    "ExplainerPageLayout",
    "ExplainerDashboard",
    "ExplainerHub",
    "JupyterExplainerDashboard",
    "ExplainerTab",
    "JupyterExplainerTab",
    "InlineExplainer",
]

import sys
import re
import json
import inspect
import requests
from typing import List, Union
from pathlib import Path
from copy import copy, deepcopy
import warnings

import oyaml as yaml

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

warnings.filterwarnings(
    "ignore",
    # NB the \n at the beginning of the message :-/
    r"\nThe dash_\w+_components package is deprecated",
    UserWarning,
    "dash_auth.plotly_auth",
)
import dash_auth

from flask import Flask, request, redirect
from flask_simplelogin import SimpleLogin, login_required
from werkzeug.security import check_password_hash, generate_password_hash

warnings.filterwarnings(
    "ignore",
    # NB the \n at the beginning of the message :-/
    r"ipykernel.comm.Comm",
    DeprecationWarning,
    "jupyter_dash.comms",
)
from jupyter_dash import JupyterDash

import plotly.io as pio

from .dashboard_methods import instantiate_component, encode_callables, decode_callables
from .dashboard_components import *
from .explainers import BaseExplainer
from . import to_html

# with pipelines we extract the final model that is fitted on raw numpy arrays and so will throw
# this error when receiving a pandas dataframe. So we suppress the warnings.
warnings.filterwarnings(
    "ignore",
    # NB the \n at the beginning of the message :-/
    r"X has feature names, but \w+ was fitted without feature names",
    UserWarning,
)


class ExplainerTabsLayout(ExplainerComponent):
    def __init__(
        self,
        explainer,
        tabs,
        title="Explicador do Modelo",  # Traduzido
        name=None,
        description=None,
        header_hide_title=False,
        header_hide_selector=False,
        header_hide_download=False,
        hide_poweredby=False,
        block_selector_callbacks=False,
        pos_label=None,
        fluid=True,
        **kwargs,
    ):
        """Generates a multi tab layout from a a list of ExplainerComponents.
        If the component is a class definition, it gets instantiated first. If
        the component is not derived from an ExplainerComponent, then attempt
        with duck typing to nevertheless instantiate a layout.

        Args:
            explainer ([type]): explainer
            tabs (list[ExplainerComponent class or instance]): list of
                ExplainerComponent class definitions or instances.
            title (str, optional): [description]. Defaults to 'Explicador do Modelo'. # Atualizado docstring (opcional)
            description (str, optional): description tooltip to add to the title.
            header_hide_title (bool, optional): Hide the title. Defaults to False.
            header_hide_selector (bool, optional): Hide the positive label selector.
                        Defaults to False.
            header_hide_download (bool, optional): Hide the download link.
                Defaults to False.
            hide_poweredby (bool, optional): hide the powered by footer
            block_selector_callbacks (bool, optional): block the callback of the
                        pos label selector. Useful to avoid clashes when you
                        have your own PosLabelSelector in your layout.
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            fluid (bool, optional): Stretch layout to fill space. Defaults to False.
        """
        super().__init__(explainer, title, name)

        if self.block_selector_callbacks:
            self.header_hide_selector = True
        self.fluid = fluid

        self.selector = PosLabelSelector(explainer, name="0", pos_label=pos_label)
        self.tabs = [
            instantiate_component(tab, explainer, name=str(i + 1), **kwargs)
            for i, tab in enumerate(tabs)
        ]
        assert (
            len(self.tabs) > 0
        ), "Ao passar uma lista para 'tabs', é necessário passar pelo menos um separador válido!" # Traduzido assert message

        self.register_components(*self.tabs)

        self.downloadable_tabs = [
            tab for tab in self.tabs if tab.to_html(add_header=False) != "<div></div>"
        ]
        if not self.downloadable_tabs:
            self.header_hide_download = True

        self.connector = PosLabelConnector(self.selector, self.tabs)

    def layout(self):
        """returns a multitab layout plus ExplainerHeader"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    html.H1(self.title, id="dashboard-title"),
                                    dbc.Tooltip(
                                        self.description, target="dashboard-title"
                                    ),
                                ],
                                width="auto",
                            ),
                            hide=self.header_hide_title,
                        ),
                        make_hideable(
                            dbc.Col([self.selector.layout()], md=3),
                            hide=self.header_hide_selector,
                        ),
                        dbc.Col([], class_name="me-auto"),
                        make_hideable(
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dcc.Download("download-page-" + self.name),
                                            dbc.DropdownMenu(
                                                [
                                                    dbc.DropdownMenuItem(
                                                        "Todos os separadores",  # Traduzido
                                                        id="download-button-all"
                                                        + self.name,
                                                        n_clicks=None,
                                                    ),
                                                    dbc.DropdownMenuItem(divider=True),
                                                    *[
                                                        dbc.DropdownMenuItem(
                                                            tab.title, # Assume que tab.title já está traduzido
                                                            id="download-button-"
                                                            + tab.name,
                                                            n_clicks=None,
                                                        )
                                                        for tab in self.downloadable_tabs
                                                    ],
                                                ],
                                                label="Descarregar", # Traduzido
                                                color="link",
                                                right=True,
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "justify-content": "flex-end",
                                        },
                                    ),
                                ],
                                md="auto",
                                className="ml-auto",
                                align="center",
                            ),
                            hide=self.header_hide_download,
                        ),
                    ],
                    justify="start",
                    style=dict(marginBottom=10),
                ),
                dcc.Tabs(
                    id="tabs",
                    value=self.tabs[0].name,
                    children=[
                        dcc.Tab(
                            label=tab.title, # Assume que tab.title já está traduzido
                            id=tab.name,
                            value=tab.name,
                            children=tab.layout(),
                        )
                        for tab in self.tabs
                    ],
                ),
                make_hideable(
                    html.Div(
                        [
                            html.Small("desenvolvido por: "), # Traduzido
                            html.Small(
                                html.A(
                                    "explainerdashboard", # Mantido nome do projeto
                                    className="text-muted",
                                    target="_blank",
                                    href="https://github.com/oegedijk/explainerdashboard",
                                )
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justify-content": "flex-end",
                            "text-align": "right",
                        },
                    ),
                    hide=self.hide_poweredby,
                ),
            ],
            fluid=self.fluid,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title) # Assume self.title traduzido
        tabs = {
            tab.title: tab.to_html(state_dict, add_header=False) for tab in self.tabs # Assume tab.title traduzido
        }
        tabs = {tab: html for tab, html in tabs.items() if html != "<div></div>"}
        html += to_html.tabs(tabs)
        if add_header:
            return to_html.add_header(html)
        return html

    def register_callbacks(self, app):
        """Registers callbacks for all tabs"""
        for tab in self.tabs:
            try:
                tab.register_callbacks(app)
            except AttributeError:
                print(f"Aviso: {tab} não possui um método register_callbacks!") # Traduzido

        if not self.block_selector_callbacks:
            if any([tab.has_pos_label_connector() for tab in self.tabs]):
                print(
                    "Aviso: detetados PosLabelConnectors já no layout. " # Traduzido
                    "Isto pode entrar em conflito com o seletor global de etiqueta pos e gerar erros de callback duplicados. " # Traduzido
                    "Se assim for, defina block_selector_callbacks=True." # Traduzido
                )
            self.connector.register_callbacks(app)

        @app.callback(
            Output("download-page-" + self.name, "data"),
            [
                Input("download-button-all" + self.name, "n_clicks"),
                *[
                    Input("download-button-" + tab.name, "n_clicks")
                    for tab in self.downloadable_tabs
                ],
            ],
            [State(id_, prop_) for id_, prop_ in self.get_state_tuples()],
        )
        def download_html(*args):
            state_dict = dict(
                zip(self.get_state_tuples(), args[1 + len(self.downloadable_tabs) :])
            )

            ctx = dash.callback_context
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if button_id == "download-button-all" + self.name:
                return dict(content=self.to_html(state_dict), filename="dashboard.html") # Mantido filename
            for tab in self.downloadable_tabs:
                if button_id == "download-button-" + tab.name:
                    return dict(
                        content=tab.to_html(state_dict), filename="dashboard.html" # Mantido filename
                    )
            raise PreventUpdate

    def calculate_dependencies(self):
        """Calculates dependencies for all tabs"""
        for tab in self.tabs:
            try:
                tab.calculate_dependencies()
            except AttributeError:
                print(f"Aviso: {tab} não possui um método calculate_dependencies!") # Traduzido


class ExplainerPageLayout(ExplainerComponent):
    def __init__(
        self,
        explainer,
        component,
        title=None,#"Explicação do Modelo", # Ajustado capitalização
        name=None,
        description=None,
        header_hide_title=False,
        header_hide_selector=False,
        header_hide_download=False,
        hide_poweredby=False,
        block_selector_callbacks=False,
        pos_label=None,
        fluid=False,
        **kwargs,
    ):
        """Generates a single page layout from a single ExplainerComponent.
        If the component is a class definition, it gets instantiated.

        If the component is not derived from an ExplainerComponent, then tries
        with duck typing to nevertheless instantiate a layout.


        Args:
            explainer ([type]): explainer
            component (ExplainerComponent class or instance): ExplainerComponent
                        class definition or instance.
            title (str, optional):  Defaults to 'Explicação do Modelo'. # Atualizado docstring
            description (str, optional): Will be displayed as title tooltip.
            header_hide_title (bool, optional): Hide the title. Defaults to False.
            header_hide_selector (bool, optional): Hide the positive label selector.
                        Defaults to False.
            header_hide_download (bool, optional): Hide the download link. Defaults to False.
            hide_poweredby (bool, optional): hide the powered by footer
            block_selector_callbacks (bool, optional): block the callback of the
                        pos label selector. Useful to avoid clashes when you
                        have your own PosLabelSelector in your layout.
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            fluid (bool, optional): Stretch layout to fill space. Defaults to False.
        """
        super().__init__(explainer, title, name)
        self.title = title

        if self.block_selector_callbacks:
            self.header_hide_selector = True
        self.fluid = fluid

        self.selector = PosLabelSelector(explainer, name="0", pos_label=pos_label)
        self.page = instantiate_component(component, explainer, name="1", **kwargs)
        self.register_components(self.page)

        self.connector = PosLabelConnector(self.selector, self.page)

        self.page_layout = self.page.layout()
        if hasattr(self.page_layout, "fluid"):
            self.fluid = self.page_layout.fluid

        if self.page.to_html(add_header=False) == "<div></div>":
            self.header_hide_download = True

    def layout(self):
        """returns single page layout with an ExplainerHeader"""
        return dbc.Container(
            [
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    html.H1(self.title, id="dashboard-title"), # Assume self.title traduzido
                                    dbc.Tooltip(
                                        self.description, target="dashboard-title" # Assume self.description traduzido
                                    ),
                                ],
                                width="auto",
                                align="start",
                            ),
                            hide=self.header_hide_title,
                        ),
                        make_hideable(
                            dbc.Col([self.selector.layout()], md=3, align="start"),
                            hide=self.header_hide_selector,
                        ),
                        make_hideable(
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "Descarregar", # Traduzido
                                                id="download-page-button-" + self.name,
                                                color="link",
                                            ),
                                            dcc.Download("download-page-" + self.name),
                                        ],
                                        style={
                                            "display": "flex",
                                            "justify-content": "flex-end",
                                        },
                                    ),
                                ],
                                md="auto",
                                className="ml-auto",
                                align="center",
                            ),
                            hide=self.header_hide_download,
                        ),
                    ],
                    justify="start",
                ),
                self.page_layout,
                dbc.Row(
                    [
                        make_hideable(
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.Small("desenvolvido por: "), # Traduzido
                                            html.Small(
                                                html.A(
                                                    "Bright", # Mantido nome próprio/marca
                                                    className="text-muted",
                                                    target="_blank",
                                                    href="https://brightcity.pt/",
                                                )
                                            ),
                                        ]
                                    ),
                                ],
                                md="3",
                            ),
                            hide=self.hide_poweredby,
                        ),
                    ],
                    justify="end",
                ),
            ],
            fluid=self.fluid,
        )

    def to_html(self, state_dict=None, add_header=True):
        html = to_html.title(self.title) # Assume self.title traduzido
        html += self.page.to_html(state_dict, add_header=False)
        if add_header:
            return to_html.add_header(html)
        return html

    def register_callbacks(self, app):
        """Register callbacks of page"""
        try:
            self.page.register_callbacks(app)
        except AttributeError:
            print(f"Aviso: {self.page} não possui um método register_callbacks!") # Traduzido
        if not self.block_selector_callbacks:
            if (
                hasattr(self.page, "has_pos_label_connector")
                and self.page.has_pos_label_connector()
            ):
                print(
                    "Aviso: detetados PosLabelConnectors já no layout. " # Traduzido
                    "Isto pode entrar em conflito com o seletor global de etiqueta pos e gerar erros de callback duplicados. " # Traduzido
                    "Se assim for, defina block_selector_callbacks=True." # Traduzido
                )
            self.connector.register_callbacks(app)

        @app.callback(
            Output("download-page-" + self.name, "data"),
            [Input("download-page-button-" + self.name, "n_clicks")],
            [State(id_, prop_) for id_, prop_ in self.page.get_state_tuples()],
        )
        def download_html(n_clicks, *args):
            if n_clicks is not None:
                state_dict = dict(zip(self.get_state_tuples(), args))
                return dict(
                    content=self.to_html(state_dict, add_header=True),
                    filename="dashboard.html", # Mantido filename
                )
            raise PreventUpdate

    def calculate_dependencies(self):
        """Calculate dependencies of page"""
        try:
            self.page.calculate_dependencies()
        except AttributeError:
            print(
                f"Aviso: {self.page} não possui um método calculate_dependencies!", # Traduzido
                flush=True,
            )


class ExplainerDashboard:
    def __init__(
        self,
        explainer: BaseExplainer = None,
        tabs: Union[ExplainerComponent, List[ExplainerComponent]] = None,
        title: str = None, #"Explicação do Modelo", # Ajustado capitalização
        name: str = None,
        description: str = None,
        simple=False,
        hide_header: bool = False,
        header_hide_title: bool = False,
        header_hide_selector: bool = False,
        header_hide_download: bool = False,
        hide_poweredby: bool = False,
        block_selector_callbacks: bool = False,
        pos_label: Union[str, int] = None,
        fluid: bool = True,
        mode: str = "dash",
        width: int = 1000,
        height: int = 800,
        bootstrap: str = None,
        external_stylesheets: List[str] = None,
        server: bool = True,
        url_base_pathname: str = None,
        routes_pathname_prefix: str = None,
        requests_pathname_prefix: str = None,
        responsive: bool = True,
        logins: List[List[str]] = None,
        port: int = 8050,
        importances: bool = True,
        model_summary: bool = True,
        contributions: bool = True,
        whatif: bool = True,
        shap_dependence: bool = True,
        shap_interaction: bool = True,
        decision_trees: bool = True,
        **kwargs,
    ):
        """Creates an explainerdashboard out of an Explainer object.


        single page dashboard:
            If tabs is a single ExplainerComponent class or instance, display it
            as a standalone page without tabs.

        Multi tab dashboard:
            If tabs is a list of ExplainerComponent classes or instances, then construct
            a layout with a tab per component. Instead of components you can also pass
            the following strings: "importances", "model_summary", "contributions",
            "shap_dependence", "shap_interaction" or "decision_trees". You can mix and
            combine these different modularities, e.g.:
                tabs=[ImportancesTab, "contributions", custom_tab]

        If tabs is None, then construct tabs based on the boolean parameters:
            importances, model_summary, contributions, shap_dependence,
            shap_interaction and decision_trees, which all default to True.

        You can select four different modes:
            - 'dash': standard dash.Dash() app
            - 'inline': JupyterDash app inline in a notebook cell output
            - 'jupyterlab': JupyterDash app in jupyterlab pane
            - 'external': JupyterDash app in external tab

        You can switch off the title and positive label selector
            with header_hide_title=True and header_hide_selector=True.

        You run the dashboard
            with e.g. ExplainerDashboard(explainer).run(port=8050)


        Args:
            explainer(): explainer object
            tabs(): single component or list of components
            title(str, optional): title of dashboard, defaults to 'Explicação do Modelo' # Atualizado docstring
            name (str, optional): name of the dashboard. Used for assigning url in ExplainerHub.
            description (str, optional): summary for dashboard. Gets used for title tooltip and
                in description for ExplainerHub.
            simple(bool, optional): instead of full dashboard with all tabs display
                a single page SimplifiedClassifierDashboard or SimplifiedRegressionDashboard.
            hide_header (bool, optional) hide the header (title+selector), defaults to False.
            header_hide_title(bool, optional): hide the title, defaults to False
            header_hide_selector(bool, optional): hide the positive class selector
                for classifier models, defaults, to False
            header_hide_download (bool, optional): hide the download link in the header.
                Defaults to False.
            hide_poweredby (bool, optional): hide the powered by footer
            block_selector_callbacks (bool, optional): block the callback of the
                        pos label selector. Useful to avoid clashes when you
                        have your own PosLabelSelector in your layout.
                        Defaults to False.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            mode(str, {'dash', 'inline' , 'jupyterlab', 'external'}, optional):
                type of dash server to start. 'inline' runs in a jupyter notebook output cell.
                'jupyterlab' runs in a jupyterlab pane. 'external' runs in an external tab
                while keeping the notebook interactive.
            fluid(bool, optional): whether to stretch the layout to available space.
                    Defaults to True.
            width(int, optional): width of notebook output cell in pixels, defaults to 1000.
            height(int, optional): height of notebookn output cell in pixels, defaults to 800.
            bootstrap (str, optional): link to bootstrap url. Can use dbc.themese
                to generate the url, e.g. bootstrap=dbc.themes.FLATLY. Defaults
                to default bootstrap theme that is stored in the /assets folder
                so that it works even behind a firewall.
            external_stylesheets(list, optional): additional external stylesheets
                to add. (for themes use the bootstrap parameter)
            server (Flask instance or bool): either an instance of an existing Flask
                server to tie the dashboard to, or True in which case a new Flask
                server is created.
            url_base_pathname (str): url_base_pathname for dashboard,
                e.g. "/dashboard". Defaults to None.
            responsive (bool):  make layout responsive to viewport size
                (i.e. reorganize bootstrap columns on small devices). Set to False
                when e.g. testing with a headless browser. Defaults to True.
            logins (list of lists): list of (hardcoded) logins, e.g.
                [['login1', 'password1'], ['login2', 'password2']].
                Defaults to None (no login required)
            importances(bool, optional): include ImportancesTab, defaults to True.
            model_summary(bool, optional): include ModelSummaryTab, defaults to True.
            contributions(bool, optional): include ContributionsTab, defaults to True.
            whatif (bool, optional): include WhatIfTab, defaults to True.
            shap_dependence(bool, optional): include ShapDependenceTab, defaults to True.
            shap_interaction(bool, optional): include InteractionsTab if model allows it, defaults to True.
            decision_trees(bool, optional): include DecisionTreesTab if model allows it, defaults to True.
        """
        print("A construir o ExplainerDashboard...", flush=True) # Traduzido

        self._store_params(no_param=["explainer", "tabs", "server"])
        self._stored_params["tabs"] = self._tabs_to_yaml(tabs)

        if not hasattr(explainer, "__version__"):
            raise ValueError(
                f"O {explainer.__class__.__name__} foi gerado " # Traduzido
                "com uma versão de explainerdashboard<0.3 e, portanto, não é " # Traduzido
                "compatível com esta versão do ExplainerDashboard devido a " # Traduzido
                "alterações significativas entre versões principais! Por favor, reconstrua " # Traduzido
                f"o seu {explainer.__class__.__name__} com esta versão, ou " # Traduzido
                "reverta para explainerdashboard==0.2.20.1!" # Traduzido
            )

        dynamic_dropdown_threshold = min(
            1000, self.kwargs.get("max_idxs_in_dropdown", float("inf"))
        )
        if (
            self.explainer is not None
            and len(self.explainer) > dynamic_dropdown_threshold
        ):
            from pkg_resources import parse_version

            if parse_version(dash.__version__) > parse_version("2.6.2"):
                print(
                    f"AVISO: o número de índices (={len(self.explainer)}) > max_idxs_in_dropdown(={dynamic_dropdown_threshold}). " # Traduzido
                    f"No entanto, com a sua versão instalada do dash({dash.__version__}) a pesquisa no dropdown pode não funcionar corretamente. " # Traduzido
                    f"Pode reverter para `pip install dash==2.6.2` que deverá funcionar melhor por agora..." # Traduzido
                )

        if self.description is None:
            self.description = """Este painel mostra o funcionamento de um modelo de 
            machine learning treinado e explica as suas previsões.""" # Traduzido
            self._stored_params["description"] = self.description

        try:
            ipython_kernel = str(get_ipython())
            self.is_notebook = True
            self.is_colab = True if "google.colab" in ipython_kernel else False
        except:
            self.is_notebook, self.is_colab = False, False

        if self.mode == "dash" and self.is_colab:
            print(
                "Detetado ambiente Google Colab, a definir mode='external'", flush=True # Traduzido
            )
            self.mode = "external"
        elif self.mode == "dash" and self.is_notebook:
            print(
                "Detetado ambiente de notebook, considere definir " # Traduzido
                "mode='external', mode='inline' ou mode='jupyterlab' " # Traduzido
                "para manter o notebook interativo enquanto o painel " # Traduzido
                "está em execução...", # Traduzido
                flush=True,
            )

        if self.bootstrap is not None:
            bootstrap_theme = (
                self.bootstrap
                if isinstance(self.bootstrap, str)
                else dbc.themes.BOOTSTRAP
            )
            if self.external_stylesheets is None:
                self.external_stylesheets = [bootstrap_theme]
            else:
                self.external_stylesheets.append(bootstrap_theme)

        self.app = self._get_dash_app()

        if logins is not None:
            if (
                len(logins) == 2
                and isinstance(logins[0], str)
                and isinstance(logins[1], str)
            ):
                self.logins = [logins]
                self._stored_params["logins"] = self.logins
            assert isinstance(self.logins, list), (
                "O parâmetro logins deve ser uma lista de listas de pares str, ex." # Traduzido
                " logins=[['utilizador1', 'senha1'], ['utilizador2', 'senha2']]!" # Traduzido
            )
            for login in self.logins:
                assert isinstance(login, list), (
                    "O parâmetro logins deve ser uma lista de listas de pares str, " # Traduzido
                    "ex. logins=[['utilizador1', 'senha1'], ['utilizador2', 'senha2']]!" # Traduzido
                )
                assert isinstance(login[0], str) and isinstance(login[1], str), (
                    "Para logins como [['utilizador1', 'senha1']] utilizador1 e " # Traduzido
                    "senha1 devem ser do tipo str!" # Traduzido
                )
            self.auth = dash_auth.BasicAuth(self.app, self.logins)
        self.app.title = title # Assume title traduzido

        assert "BaseExplainer" in str(explainer.__class__.mro()), (
            "explainer deve ser uma instância de BaseExplainer, como " # Traduzido
            "ClassifierExplainer ou RegressionExplainer!" # Traduzido
        )

        if tabs is None:
            if simple:
                self.header_hide_selector = True
                if explainer.is_classifier:
                    tabs = SimplifiedClassifierComposite(
                        explainer, title=self.title, hide_title=True, **kwargs
                    )
                else:
                    tabs = SimplifiedRegressionComposite(
                        explainer, title=self.title, hide_title=True, **kwargs
                    )
            else:
                tabs = []
                if model_summary and explainer.y_missing:
                    print(
                        "Nenhuma etiqueta y foi passada para o Explainer, a definir" # Traduzido
                        " model_summary=False...", # Traduzido
                        flush=True,
                    )
                    model_summary = False
                if shap_interaction and (not explainer.interactions_should_work):
                    print(
                        "Para este tipo de modelo e model_output as interações não " # Traduzido
                        "funcionam, a definir shap_interaction=False...", # Traduzido
                        flush=True,
                    )
                    shap_interaction = False
                if decision_trees and not hasattr(explainer, "is_tree_explainer"):
                    print(
                        "O objeto explainer não tem a propriedade decision_trees. Logo " # Traduzido
                        "a definir decision_trees=False...", # Traduzido
                        flush=True,
                    )
                    decision_trees = False

                if importances:
                    tabs.append(ImportancesComposite)
                if model_summary:
                    tabs.append(
                        ClassifierModelStatsComposite
                        if explainer.is_classifier
                        else RegressionModelStatsComposite
                    )
                if contributions:
                    tabs.append(IndividualPredictionsComposite)
                if whatif:
                    tabs.append(WhatIfComposite)
                if shap_dependence:
                    tabs.append(ShapDependenceComposite)
                if shap_interaction:
                    print(
                        "Aviso: calcular valores de interação SHAP pode ser lento! " # Traduzido
                        "Passe shap_interaction=False para remover o separador de interações.", # Traduzido
                        flush=True,
                    )
                    tabs.append(ShapInteractionsComposite)
                if decision_trees:
                    tabs.append(DecisionTreesComposite)

        if isinstance(tabs, list) and len(tabs) == 1:
            tabs = tabs[0]

        if self.hide_header:
            self.header_hide_title = True
            self.header_hide_selector = True
            self.header_hide_download = True

        print("A gerar layout...") # Traduzido
        _, i = yield_id(return_i=True)  # store id generator index
        reset_id_generator("db")  # reset id generator to 0 with prefix "db"
        if hasattr(self.explainer, "_index_list"):
            del (
                self.explainer._index_list
            )  # delete cached ._index_list to force re-download

        if isinstance(tabs, list):
            tabs = [self._convert_str_tabs(tab) for tab in tabs]
            self.explainer_layout = ExplainerTabsLayout(
                explainer,
                tabs,
                title, # Assume title traduzido
                description=self.description, # Assume description traduzido
                **update_kwargs(
                    kwargs,
                    header_hide_title=self.header_hide_title,
                    header_hide_selector=self.header_hide_selector,
                    header_hide_download=self.header_hide_download,
                    hide_poweredby=self.hide_poweredby,
                    block_selector_callbacks=self.block_selector_callbacks,
                    pos_label=self.pos_label,
                    fluid=fluid,
                ),
            )
        else:
            tabs = self._convert_str_tabs(tabs)
            self.explainer_layout = ExplainerPageLayout(
                explainer,
                tabs,
                title, # Assume title traduzido
                description=self.description, # Assume description traduzido
                **update_kwargs(
                    kwargs,
                    header_hide_title=self.header_hide_title,
                    header_hide_selector=self.header_hide_selector,
                    header_hide_download=self.header_hide_download,
                    hide_poweredby=self.hide_poweredby,
                    block_selector_callbacks=self.block_selector_callbacks,
                    pos_label=self.pos_label,
                    fluid=self.fluid,
                ),
            )

        self.app.layout = self.explainer_layout.layout()
        reset_id_generator(start=i + 1)  # reset id generator to previous index

        print("A calcular dependências...", flush=True) # Traduzido
        self.explainer_layout.calculate_dependencies()
        print(
            "Lembrete: pode guardar o explainer (incluindo dependências calculadas) " # Traduzido
            "com explainer.dump('explainer.joblib') e " # Traduzido
            "recarregar com, por exemplo, ClassifierExplainer.from_file('explainer.joblib')", # Traduzido
            flush=True,
        )
        print("A registar callbacks...", flush=True) # Traduzido
        self.explainer_layout.register_callbacks(self.app)

    def to_html(self):
        """return static html output of dashboard"""
        return self.explainer_layout.to_html()

    def save_html(self, filename: Union[str, Path] = None):
        """Store output of to_html to a file

        Args:
            filename (str, Path): filename to store html
        """
        html = self.to_html()
        if filename is None:
            return html
        with open(filename, "w") as f:
            f.write(html)

    @classmethod
    def from_config(cls, arg1, arg2=None, **update_params):
        """Loading a dashboard from a configuration .yaml file. You can either
        pass both an explainer and a yaml file generated with
        ExplainerDashboard.to_yaml("dashboard.yaml"):

          db = ExplainerDashboard.from_config(explainer, "dashboard.yaml")

        When you specify an explainerfile in to_yaml with
        ExplainerDashboard.to_yaml("dashboard.yaml", explainerfile="explainer.joblib"),
        you can also pass just the .yaml:

          db = ExplainerDashboard.from_config("dashboard.yaml")

        You can also load the explainerfile seperately:

          db = ExplainerDashboard.from_config("explainer.joblib", "dashboard.yaml")

        Args:
            arg1 (explainer or config): arg1 should either be a config (yaml or dict),
                or an explainer (instance or str/Path).
            arg2 ([type], optional): If arg1 is an explainer, arg2 should be config.
            update_params (dict): You can override parameters in the the yaml
                config by passing additional kwargs to .from_config()

        Returns:
            ExplainerDashboard
        """
        if arg2 is None:
            if isinstance(arg1, (Path, str)) and str(arg1).endswith(".yaml"):
                config = yaml.safe_load(open(str(arg1), "r"))
            elif isinstance(arg1, dict):
                config = arg1
                assert (
                    "dashboard" in config
                ), "O ficheiro .yaml não tem o parâmetro `dashboard`." # Traduzido
                assert (
                    "explainerfile" in config["dashboard"]
                ), "O ficheiro .yaml não tem o parâmetro explainerfile" # Traduzido

            explainer = BaseExplainer.from_file(config["dashboard"]["explainerfile"])
        else:
            if isinstance(arg1, BaseExplainer):
                explainer = arg1
            elif isinstance(arg1, (Path, str)) and (
                str(arg1).endswith(".joblib")
                or str(arg1).endswith(".pkl")
                or str(arg1).endswith(".dill")
            ):
                explainer = BaseExplainer.from_file(arg1)
            else:
                raise ValueError(
                    "Ao passar dois argumentos para ExplainerDashboard.from_config(arg1, arg2), " # Traduzido
                    "arg1 deve ser um explainer ou um nome de ficheiro de explainer (ex: 'explainer.joblib')!" # Traduzido
                )
            if isinstance(arg2, (Path, str)) and str(arg2).endswith(".yaml"):
                config = yaml.safe_load(open(str(arg2), "r"))
            elif isinstance(arg2, dict):
                config = arg2
            else:
                raise ValueError(
                    "Ao passar dois argumentos para ExplainerDashboard.from_config(arg1, arg2), " # Traduzido
                    "arg2 deve ser um ficheiro .yaml ou um dict!" # Traduzido
                )

        dashboard_params = decode_callables(config["dashboard"]["params"])

        for k, v in update_params.items():
            if k in dashboard_params:
                dashboard_params[k] = v
            elif "kwargs" in dashboard_params:
                dashboard_params["kwargs"][k] = v
            else:
                dashboard_params["kwargs"] = dict(k=v)

        if "kwargs" in dashboard_params:
            kwargs = dashboard_params.pop("kwargs")
        else:
            kwargs = {}

        if "tabs" in dashboard_params:
            tabs = cls._yamltabs_to_tabs(dashboard_params["tabs"], explainer)
            del dashboard_params["tabs"]
            return cls(explainer, tabs, **dashboard_params, **kwargs)
        else:
            return cls(explainer, **dashboard_params, **kwargs)

    def to_yaml(
        self,
        filepath: Union[str, Path] = None,
        return_dict: bool = False,
        explainerfile: str = "explainer.joblib",
        dump_explainer: bool = False,
        explainerfile_absolute_path: Union[str, Path, bool] = None,
    ):
        """Returns a yaml configuration of the current ExplainerDashboard
        that can be used by the explainerdashboard CLI or to reinstate an identical
        dashboard from the (dumped) explainer and saved configuration. Recommended filename
        is `dashboard.yaml`.

        Args:
            filepath ({str, Path}, optional): Filepath to dump yaml. If None
                returns the yaml as a string. Defaults to None.
            return_dict (bool, optional): instead of yaml return dict with
                config.
            explainerfile (str, optional): filename of explainer dump. Defaults
                to `explainer.joblib`. Should en in either .joblib .dill or .pkl
            dump_explainer (bool, optional): dump the explainer along with the yaml.
                You must pass explainerfile parameter for the filename. Defaults to False.
            explainerfile_absolute_path (str, Path, bool, optional): absolute path to save explainerfile if
                not in the same directory as the yaml file. You can also pass True which still
                saves the explainerfile to the filepath directory, but adds an absolute path
                to the yaml file.

        """
        import oyaml as yaml

        dashboard_config = dict(
            dashboard=dict(explainerfile=str(explainerfile), params=self._stored_params)
        )

        if return_dict:
            return dashboard_config

        if filepath is not None:
            dashboard_path = Path(filepath).absolute().parent
            dashboard_path.mkdir(parents=True, exist_ok=True)

            if explainerfile_absolute_path:
                if isinstance(explainerfile_absolute_path, bool):
                    explainerfile_absolute_path = dashboard_path / explainerfile
                dashboard_config["dashboard"]["explainerfile"] = str(
                    Path(explainerfile_absolute_path).absolute()
                )
            else:
                explainerfile_absolute_path = dashboard_path / explainerfile

            print(
                f"A guardar configuração .yaml para {Path(filepath).absolute()}...", # Traduzido
                flush=True,
            )
            yaml.dump(dashboard_config, open(filepath, "w"))

            if dump_explainer:
                print(
                    f"A guardar explainer para {explainerfile_absolute_path}...", flush=True # Traduzido
                )
                self.explainer.dump(explainerfile_absolute_path)
            return
        return yaml.dump(dashboard_config)

    def _store_params(self, no_store=None, no_attr=None, no_param=None):
        """Stores the parameter of the class to instance attributes and
        to a ._stored_params dict. You can optionall exclude all or some
        parameters from being stored.

        Args:
            no_store ({bool, List[str]}, optional): If True do not store any
                parameters to either attribute or _stored_params dict. If
                a list of str, then do not store parameters with those names.
                Defaults to None.
            no_attr ({bool, List[str]},, optional): . If True do not store any
                parameters to class attribute. If
                a list of str, then do not store parameters with those names.
                Defaults to None.
            no_param ({bool, List[str]},, optional): If True do not store any
                parameters to _stored_params dict. If
                a list of str, then do not store parameters with those names.
                Defaults to None.
        """
        if not hasattr(self, "_stored_params"):
            self._stored_params = {}

        frame = sys._getframe(1)
        args = frame.f_code.co_varnames[1 : frame.f_code.co_argcount]
        args_dict = {arg: frame.f_locals[arg] for arg in args}

        if "kwargs" in frame.f_locals:
            args_dict["kwargs"] = frame.f_locals["kwargs"]

        if isinstance(no_store, bool) and no_store:
            return
        else:
            if no_store is None:
                no_store = tuple()

        if isinstance(no_attr, bool) and no_attr:
            dont_attr = True
        else:
            if no_attr is None:
                no_attr = tuple()
            dont_attr = False

        if isinstance(no_param, bool) and no_param:
            dont_param = True
        else:
            if no_param is None:
                no_param = tuple()
            dont_param = False

        for name, value in args_dict.items():
            if not dont_attr and name not in no_store and name not in no_attr:
                setattr(self, name, value)
            if not dont_param and name not in no_store and name not in no_param:
                self._stored_params[name] = value

        self._stored_params = encode_callables(self._stored_params)

    def _convert_str_tabs(self, component):
        # Estas strings são chaves internas, não devem ser traduzidas aqui.
        # A tradução acontece no título do componente que estas chaves instanciam.
        if isinstance(component, str):
            if component == "importances":
                return ImportancesComposite
            elif component == "model_summary":
                if self.explainer.is_classifier:
                    return ClassifierModelStatsComposite
                else:
                    return RegressionModelStatsComposite
            elif component == "contributions":
                return IndividualPredictionsComposite
            elif component == "whatif":
                return WhatIfComposite
            elif component == "shap_dependence":
                return ShapDependenceComposite
            elif component == "shap_interaction":
                return ShapInteractionsComposite
            elif component == "decision_trees":
                return DecisionTreesComposite
        return component

    @staticmethod
    def _tabs_to_yaml(tabs):
        """converts tabs to a yaml friendly format"""
        if tabs is None:
            return None

        def get_name_and_module(component):
            if inspect.isclass(component) and issubclass(component, ExplainerComponent):
                return dict(
                    name=component.__name__, module=component.__module__, params=None
                )
            elif isinstance(component, ExplainerComponent):
                component_imports = dict(component.component_imports)
                del component_imports[component.__class__.__name__]
                return dict(
                    name=component.__class__.__name__,
                    module=component.__class__.__module__,
                    params=component._stored_params,
                    component_imports=component_imports,
                )
            else:
                raise ValueError(
                    f"Por favor, passe apenas strings ou ExplainerComponents para o parâmetro `tabs`!" # Traduzido
                    f"Você passou {component.__class__}" # Traduzido
                )

        if not hasattr(tabs, "__iter__"):
            return tabs if isinstance(tabs, str) else get_name_and_module(tabs)

        return [
            tab if isinstance(tab, str) else get_name_and_module(tab) for tab in tabs
        ]

    @staticmethod
    def _yamltabs_to_tabs(yamltabs, explainer):
        """converts a yaml tabs list back to ExplainerDashboard compatible original"""
        from importlib import import_module

        if yamltabs is None:
            return None

        def instantiate_tab(tab, explainer, name=None):
            if isinstance(tab, str):
                return tab
            elif isinstance(tab, dict):
                # print(tab) # Mantido para debug, se necessário
                if "component_imports" in tab and tab["component_imports"] is not None:
                    for class_name, module_name in tab["component_imports"].items():
                        if class_name not in globals():
                            import_module(class_module, class_name)
                tab_class = getattr(import_module(tab["module"]), tab["name"])
                if tab["params"] is None:
                    return tab_class
                else:
                    if not "name" in tab["params"] or tab["params"]["name"] is None:
                        tab["params"]["name"] = name

                    tab["params"] = decode_callables(tab["params"])
                    tab_instance = tab_class(explainer, **tab["params"])
                    return tab_instance
            else:
                raise ValueError(
                    "o separador yaml deve ser uma string, ex: 'importances', " # Traduzido
                    "ou um dict(name=..,module=..,params=...)" # Traduzido
                )

        if not hasattr(yamltabs, "__iter__"):
            return instantiate_tab(yamltabs, explainer, name="1")
        tabs = [
            instantiate_tab(tab, explainer, name=str(i + 1))
            for i, tab in enumerate(yamltabs)
        ]
        # print(tabs) # Mantido para debug, se necessário
        return tabs

    def _get_dash_app(self):
        if self.responsive:
            meta_tags = [
                {
                    "name": "viewport",
                    "content": "width=device-width, initial-scale=1.0, maximum-scale=1.2, minimum-scale=0.5,",
                }
            ]
        else:
            meta_tags = None

        if self.bootstrap is not None:
            assets_ignore = "^bootstrap.min.css$"
        else:
            assets_ignore = ""
        if self.mode == "dash":
            app = dash.Dash(
                __name__,
                server=self.server,
                external_stylesheets=self.external_stylesheets,
                assets_ignore=assets_ignore,
                url_base_pathname=self.url_base_pathname,
                routes_pathname_prefix=self.routes_pathname_prefix,
                requests_pathname_prefix=self.requests_pathname_prefix,
                meta_tags=meta_tags,
            )
        elif self.mode in ["inline", "jupyterlab", "external"]:
            app = JupyterDash(
                __name__,
                external_stylesheets=self.external_stylesheets,
                assets_ignore=assets_ignore,
                meta_tags=meta_tags,
            )
        else:
            raise ValueError(
                f"mode=={self.mode} mas deveria estar em " # Traduzido
                "{'dash', 'inline', 'juypyterlab', 'external'}" # Traduzido
            )
        app.config["suppress_callback_exceptions"] = True
        app.scripts.config.serve_locally = True
        app.css.config.serve_locally = True
        return app

    def flask_server(self):
        """returns self.app.server so that it can be exposed to e.g. gunicorn"""
        if self.mode != "dash":
            print("Aviso: em produção deve provavelmente usar mode='dash'...") # Traduzido
        return self.app.server

    def run(self, port=None, host="0.0.0.0", use_waitress=False, mode=None, **kwargs):
        """Start ExplainerDashboard on port

        Args:
            port (int, optional): port to run on. If None, then use self.port.
            host (str, optional): host to run on. Defaults to '0.0.0.0'.
            use_waitress (bool, optional): use the waitress python web server
                instead of the flask development server. Only works with mode='dash'.
                Defaults to False.
            mode(str, {'dash', 'inline' , 'jupyterlab', 'external'}, optional):
                Type of dash server to start. 'inline' runs in a jupyter notebook output cell.
                'jupyterlab' runs in a jupyterlab pane. 'external' runs in an external tab
                while keeping the notebook interactive. 'dash' is the default server.
                Overrides self.mode, in which case the dashboard will get
                rebuilt before running it with the right type of dash server.
                (dash.Dash or JupyterDash). Defaults to None (i.e. self.mode)
            Defaults to None.self.port defaults to 8050.

        Raises:
            ValueError: if mode is unknown

        """

        pio.templates.default = "none"
        if port is None:
            port = self.port
        if mode is None:
            mode = self.mode

        if use_waitress and mode != "dash":
            print(
                f"Aviso: waitress não funciona com mode={self.mode}, " # Traduzido
                "a usar o servidor JupyterDash em vez disso!", # Traduzido
                flush=True,
            )
        if mode == "dash":
            if self.mode != "dash":
                print(
                    "Aviso: ExplainerDashboard original não foi inicializado " # Traduzido
                    "com mode='dash'. A reconstruir o painel antes de iniciar:", # Traduzido
                    flush=True,
                )
                app = ExplainerDashboard.from_config(
                    self.explainer, self.to_yaml(return_dict=True), mode="dash"
                ).app
            else:
                app = self.app

            print(
                f"A iniciar ExplainerDashboard em http://{get_local_ip_adress()}:{port}", # Traduzido
                flush=True,
            )
            if use_waitress:
                from waitress import serve

                serve(app.server, host=host, port=port)
            else:
                app.run_server(port=port, host=host, **kwargs)
        else:
            if self.mode == "dash":
                print(
                    "Aviso: ExplainerDashboard original foi inicializado " # Traduzido
                    "com mode='dash'. A reconstruir o painel antes de iniciar:", # Traduzido
                    flush=True,
                )
                app = ExplainerDashboard.from_config(
                    self.explainer, self.to_yaml(return_dict=True), mode=mode
                ).app
            else:
                app = self.app
            if mode == "external":
                if not self.is_colab or self.mode == "external":
                    print(
                        f"A iniciar ExplainerDashboard em http://{get_local_ip_adress()}:{port}\n" # Traduzido
                        "Pode terminar o painel com " # Traduzido
                        f"ExplainerDashboard.terminate({port})", # Traduzido
                        flush=True,
                    )
                app.run_server(port=port, mode=mode, **kwargs)
            elif mode in ["inline", "jupyterlab"]:
                print(
                    f"A iniciar ExplainerDashboard inline (termine com " # Traduzido
                    f"ExplainerDashboard.terminate({port}))", # Traduzido
                    flush=True,
                )
                app.run_server(
                    port=port, mode=mode, width=self.width, height=self.height, **kwargs
                )
            else:
                raise ValueError(f"Modo desconhecido: mode='{mode}'!") # Traduzido

    @classmethod
    def terminate(cls, port, token=None):
        """
        Classmethodd to terminate any JupyterDash dashboard (so started with
        mode='inline',  mode='external' or mode='jupyterlab') from any
        ExplainerDashboard by specifying the right port.

        Example:
            ExplainerDashboard(explainer, mode='external').run(port=8050)

            ExplainerDashboard.terminate(8050)

        Args:
            port (int): port on which the dashboard is running.
            token (str, optional): JupyterDash._token class property.
                Defaults to the _token of the JupyterDash in the current namespace.

        Raises:
            ValueError: if can't find the port to terminate.
        """
        if token is None:
            token = JupyterDash._token

        shutdown_url = f"http://localhost:{port}/_shutdown_{token}"
        print(f"A tentar encerrar o painel na porta {port}...") # Traduzido
        try:
            response = requests.get(shutdown_url)
        except Exception as e:
            print(f"Algo parece ter falhado: {e}") # Traduzido


class ExplainerHub:
    """ExplainerHub is a way to host multiple dashboards in a single point,
    and manage access through adding user accounts.

    Example:
        ``hub = ExplainerHub([db1, db2], logins=[['user', 'password']], secret_key="SECRET")``
        ``hub.run()``


    A frontend is hosted at e.g. ``localhost:8050``, with summaries and links to
    each individual dashboard. Each ExplainerDashboard is hosted on its own url path,
    so that you can also find it directly, e.g.:
        ``localhost:8050/dashboards/dashboard1`` and ``localhost:8050/dashboards/dashboard2``.

    You can store the hub configuration, dashboard configurations, explainers
    and user database with a single command: ``hub.to_yaml('hub.yaml')``.

    You can restore the hub with ``hub2 = ExplainerHub.from_config('hub.yaml')``

    You can start the hub from the command line using the ``explainerhub`` CLI
    command: ``$ explainerhub run hub.yaml``. You can also use the CLI to
    add and delete users.

    """

    __reserved_names = {"login", "logout", "admin", "index", "hub"}

    def __init__(
        self,
        dashboards: List[ExplainerDashboard],
        title: str = "Hub de Explainers", # Traduzido (alternativa: manter "ExplainerHub")
        description: str = None,
        masonry: bool = False,
        n_dashboard_cols: int = 3,
        users_file: str = "users.yaml",
        user_json=None,
        logins: List[List] = None,
        db_users: dict = None,
        dbs_open_by_default: bool = False,
        port: int = 8050,
        min_height: int = 3000,
        secret_key: str = None,
        no_index: bool = False,
        bootstrap: str = None,
        fluid: bool = True,
        base_route: str = "dashboards",
        index_to_base_route: bool = False,
        static_to_base_route: bool = False,
        max_dashboards: int = None,
        add_dashboard_route: bool = False,
        add_dashboard_pattern: str = None,
        **kwargs,
    ):
        """

        Note:
            Logins can be defined in multiple places: users.json, ExplainerHub.logins
            and ExplainerDashboard.logins for each dashboard in dashboards.
            When users with the same username are defined in multiple
            locations then passwords are looked up in the following order:
            hub.logins > dashboard.logins > user.json

        Note:
            **kwargs will be forwarded to each dashboard in dashboards.

        Args:
            dashboards (List[ExplainerDashboard]): list of ExplainerDashboard to
                include in ExplainerHub.
            title (str, optional): title to display. Defaults to "Hub de Explainers". # Atualizado docstring
            description (str, optional): Short description of ExplainerHub.
                Defaults to default text.
            masonry (bool, optional): Lay out dashboard cards in fluid bootstrap
                masonry responsive style. Defaults to False.
            n_dashboard_cols (int, optional): If masonry is False, organize cards
                in rows and columns. Defaults to 3 columns.
            users_file (Path, optional): a .yaml or .json file used to store user and (hashed)
                password data. Defaults to 'users.yaml'.
            user_json (Path, optional): Deprecated! A .json file used to store user and (hashed)
                password data. Defaults to 'users.json'. Was replaced by users_file which
                can also be a more readable .yaml.
            logins (List[List[str, str]], optional): List of ['login', 'password'] pairs,
                e.g. logins = [['user1', 'password1'], ['user2', 'password2']]
            db_users (dict, optional): dictionary limiting access to certain
                dashboards to a subset of users, e.g
                dict(dashboard1=['user1', 'user2'], dashboard2=['user3']).
            dbs_open_by_default (bool, optional): Only force logins for dashboard
                with defined db_users. All other dashboards and index no login
                required. Default to False,
            port (int, optional): Port to run hub on. Defaults to 8050.
            min_height (int, optional) size of the iframe the holds the dashboard.
                Defaults to 3000 pixels.
            secret_key (str): Flask secret key to pass to dashboard in order to persist
                logins. Defaults to a new random uuid string every time you start
                the dashboard. (i.e. no persistence) You should store the secret
                key somewhere save, e.g. in a environmental variable.
            no_index (bool, optional): do not add the "/" route and "dashboards/_dashboard1"
                etc routes, but only mount the dashboards on e.g. dashboards/dashboard1. This
                allows you to add your own custom front_end.
            bootstrap (str, optional): url with custom bootstrap css, e.g.
                bootstrap=dbc.themes.FLATLY. Defaults to static bootstrap css.
            fluid (bool, optional): Let the bootstrap container fill the entire width
                of the browser. Defaults to True.
            base_route (str, optional): Base route for dashboard : /<base_route>/dashboard1.
                Defaults to "dashboards".
            index_to_base_route (bool, optional): Dispatches Hub to "/base_route/index" instead of the default
                "/" and "/index". Useful when the host root is not reserved for the ExplainerHub
            static_to_base_route(bool, optional): Dispatches Hub to "/base_route/static" instead of the default
                "/static". Useful when the host root is not reserved for the ExplainerHub
            max_dashboards (int, optional): Max number of dashboards in the hub. Defaults to None
                (for no limitation). If set and you add an additional dashboard, the
                first dashboard in self.dashboards will be deleted!
            add_dashboard_route (bool, optional): open a route /add_dashboard and
                /remove_dashboard
                If a user navigates to e.g. /add_dashboard/dashboards/dashboard4.yaml,
                the hub will check if there exists a folder dashboards which contains
                a dashboard4.yaml file. If so load this dashboard
                and add it to the hub. You can remove it with e.g. /remove_dashboard/dashboard4
                Alternatively you can specify a path pattern with add_dashboard_pattern.
                Warning: this will only work if you run the hub on a single worker
                or node!
            add_dashboard_pattern (str, optional): a str pattern with curly brackets
                in the place of where the dashboard.yaml file can be found. So e.g.
                if you keep your dashboards in a subdirectory dashboards with
                a subdirectory for the dashboard name and each yaml file
                called dashboard.yaml, you could set this to "dashboards/{}/dashboard.yaml",
                and then navigate to /add_dashboard/dashboard5 to add
                dashboards/dashboard5/dashboard.yaml.
            **kwargs: all kwargs will be forwarded to the constructors of
                each dashboard in dashboards dashboards.
        """
        self._store_params(no_store=["dashboards", "logins", "secret_key"])

        if user_json is not None:
            print(
                "Aviso: user_json está obsoleto, use o parâmetro users_file!" # Traduzido
            )
            self.users_file = user_json
        if self.description is None:
            self.description = (
                "Este Hub de Explainers mostra vários ExplainerDashboards.\n" # Traduzido
                "Cada painel torna transparente e explicável o funcionamento interno e as previsões de um modelo de machine " # Traduzido
                "learning treinado." # Traduzido
            )
            self._stored_params["description"] = self.description

        if (
            logins is not None
            and len(logins) == 2
            and isinstance(logins[0], str)
            and isinstance(logins[1], str)
        ):
            # if logins=['user', 'password'] then add the outer list
            logins = [logins]
        self.logins = self._hash_logins(logins)

        self.db_users = db_users if db_users is not None else {}
        self._validate_users_file(self.users_file)

        static_url_path = f"/{base_route}/static" if static_to_base_route else None
        self.app = Flask(__name__, static_url_path=static_url_path)

        if secret_key is not None:
            self.app.config["SECRET_KEY"] = secret_key
        SimpleLogin(self.app, login_checker=self._validate_user)

        assert (
            self.max_dashboards is None or len(dashboards) <= self.max_dashboards
        ), f"Deveria haver menos de {self.max_dashboards} painéis no hub." # Traduzido

        self.dashboards = self._instantiate_dashboards(dashboards, **kwargs)
        self.added_dashboard_counter = len(self.dashboards)

        self.dashboard_names = [db.name for db in self.dashboards]
        self.removed_dashboard_names = []

        if self.add_dashboard_route:
            print(
                "AVISO: se add_dashboard_route, novos painéis serão adicionados " # Traduzido
                "a uma instância/worker/nó específico do hub. Logo, isto só " # Traduzido
                "funcionará se executar o hub como um único worker num único nó!" # Traduzido
            )

        assert len(set(self.dashboard_names)) == len(
            self.dashboard_names
        ), f"Todas as propriedades .name dos painéis devem ser únicas, mas foram recebidas as seguintes: {self.dashboard_names}" # Traduzido
        illegal_names = list(set(self.dashboard_names) & self.__reserved_names)
        assert (
            not illegal_names
        ), f"As seguintes propriedades .name para painéis não são permitidas: {illegal_names}!" # Traduzido

        if self.users:
            for dashboard in self.dashboards:
                if (
                    not self.dbs_open_by_default
                    or dashboard.name in self.dashboards_with_users
                ):
                    self._protect_dashviews(
                        dashboard.app, username=self.get_dashboard_users(dashboard.name)
                    )
        if not self.no_index:
            if index_to_base_route:
                self.hub_base_url = f"/{self.base_route}/index/"
                self.index_route = f"/{self.base_route}/hub/"
            else:
                self.hub_base_url = "/index/"
                self.index_route = "/"

            self.index_page = self._get_index_page()
            if self.users and not self.dbs_open_by_default:
                self._protect_dashviews(self.index_page)
            self._add_flask_routes(self.app)

    def remove_dashboard(self, dashboard_name):
        """Remove a dashboard from the hub"""

        if dashboard_name not in self.dashboard_names:
            raise ValueError(
                f"{dashboard_name} não é um nome de painel existente. Não é possível removê-lo!" # Traduzido
            )

        index_dashboard = self.dashboard_names.index(dashboard_name)

        del self.dashboards[index_dashboard]
        del self.dashboard_names[index_dashboard]
        self.removed_dashboard_names.append(dashboard_name)
        # Remove dashboard from index
        if not self.no_index:
            self.index_page = self._get_index_page()

            if self.users and not self.dbs_open_by_default:
                self._protect_dashviews(self.index_page)

    def add_dashboard(self, dashboard: ExplainerDashboard, **kwargs):
        """Add a dashboard to the hub

        Args:
            dashboard (ExplainerDashboard): an ExplainerDashboard to add to the hub
            **kwargs: all kwargs will be forwarded to the constructors of the dashboard
        """
        # Remove first dashboard if the max_dashboard is reached
        if (
            self.max_dashboards is not None
            and len(self.dashboards) >= self.max_dashboards
        ):
            print(
                f"Aviso: excedido max_dashboards={self.max_dashboards}, a eliminar " # Traduzido
                f"o primeiro {self.base_route}/{self.dashboard_names[0]}!", # Traduzido
                flush=True,
            )
            self.remove_dashboard(self.dashboard_names[0])

        # If the dashboard has no name we give it one
        if dashboard.name is None:
            while f"dashboard{self.added_dashboard_counter}" in (
                self.dashboard_names + self.removed_dashboard_names
            ):
                self.added_dashboard_counter += 1
            dashboard.name = f"dashboard{self.added_dashboard_counter}"
        # if the dashboard name already exists we increment it by a counter
        elif dashboard.name in (self.dashboard_names + self.removed_dashboard_names):
            counter = 2
            while f"{dashboard.name}_{counter}" in (
                self.dashboard_names + self.removed_dashboard_names
            ):
                counter += 1
            dashboard.name = f"{dashboard.name}_{counter}"

        # Dashboard name should not be a reserved name
        if dashboard.name in self.__reserved_names:
            raise ValueError(
                f"As seguintes propriedades .name para painéis não são permitidas: {dashboard.name}!" # Traduzido
            )

        # If the dashboard name is unkown we create it
        update_params = dict(
            server=self.app,
            name=dashboard.name,
            url_base_pathname=f"/{self.base_route}/{dashboard.name}/",
            mode="dash",
        )
        if dashboard.logins is not None:
            for user, password in dashboard.logins:
                if user not in self.logins:
                    self.add_user(user, password)
                else:
                    print(
                        f"Aviso: {user} em {dashboard.name} já existe nos logins do " # Traduzido
                        "ExplainerHub! Não será adicionado aos logins..." # Traduzido
                    )
                self.add_user_to_dashboard(dashboard.name, user)
        config = deepcopy(dashboard.to_yaml(return_dict=True))
        config["dashboard"]["params"]["logins"] = None

        self.dashboards.append(
            ExplainerDashboard.from_config(
                dashboard.explainer, config, **update_kwargs(kwargs, **update_params)
            )
        )

        self.dashboard_names.append(dashboard.name)

        if self.users and (
            not self.dbs_open_by_default or dashboard.name in self.dashboards_with_users
        ):
            self._protect_dashviews(
                dashboard.app, username=self.get_dashboard_users(dashboard.name)
            )

        if not self.no_index:
            self.index_page = self._get_index_page()
            if self.users and not self.dbs_open_by_default:
                self._protect_dashviews(self.index_page)

            def dashboard_route(dashboard):
                def inner():
                    return self._hub_page(f"/{self.base_route}/{dashboard.name}/")

                inner.__name__ = "return_dashboard_" + dashboard.name
                return inner

        if self.users:
            self.app.route(f"/{self.base_route}/_{dashboard.name}")(
                login_required(dashboard_route(dashboard))
            )
        else:
            self.app.route(f"/{self.base_route}/_{dashboard.name}")(
                dashboard_route(dashboard)
            )
        return dashboard.name

    @classmethod
    def from_config(cls, config: Union[dict, str, Path], **update_params):
        """Instantiate an ExplainerHub based on a config file.

        Args:
            config (Union[dict, str, Path]): either a dict or a .yaml config
                file to load
            update_params: additional kwargs to override stored settings.

        Returns:
            ExplainerHub: new instance of ExplainerHub according to the config.
        """
        if isinstance(config, (Path, str)) and str(config).endswith(".yaml"):
            filepath = Path(config).parent
            config = yaml.safe_load(open(str(Path(config)), "r"))
        elif isinstance(config, dict):
            config = deepcopy(config)

        assert (
            "explainerhub" in config
        ), "YAML malformado: o ficheiro yaml explainerhub deve começar com 'explainerhub:'!" # Traduzido

        config = config["explainerhub"]

        def convert_db(db, filepath=None):
            if isinstance(db, dict):
                return db
            elif Path(db).is_absolute():
                return Path(db)
            else:
                filepath = Path(filepath or Path.cwd())
                return filepath / db

        dashboards = [
            ExplainerDashboard.from_config(convert_db(db, filepath))
            for db in config["dashboards"]
        ]
        del config["dashboards"]
        config.update(config.pop("kwargs"))
        return cls(dashboards, **update_kwargs(config, **update_params))

    def to_yaml(
        self,
        filepath: Path = None,
        dump_explainers=True,
        return_dict=False,
        integrate_dashboard_yamls=False,
        pickle_type="joblib",
    ):
        """Store ExplainerHub to configuration .yaml, store the users to users.json
        and dump the underlying dashboard .yamls and explainers.

        If filepath is None, does not store yaml config to file, but simply
        return config yaml string.

        If filepath provided and dump_explainers=True, then store all underlying
        explainers to disk.

        Args:
            filepath (Path, optional): .yaml file filepath. Defaults to None.
            dump_explainers (bool, optional): Store the explainers to disk
                along with the .yaml file. Defaults to True.
            return_dict (bool, optional): Instead of returning or storing yaml
                return a configuration dictionary. Returns a single dict as if
                separate_dashboard_yamls=True. Defaults to False.
            integrate_dashboard_yamls(bool, optional): Do not generate an individual
                .yaml file for each dashboard, but integrate them in hub.yaml.
            pickle_type ({'joblib', 'dill', 'pkl'}, optional). Format to dump explainers in.
                Defaults to "joblib". Alternatives are "dill" and "pkl".

        Returns:
            {dict, yaml, None}
        """
        filepath = Path(filepath)

        self._dump_all_users_to_file(filepath.parent / str(self.users_file))

        if filepath is None or return_dict or integrate_dashboard_yamls:
            hub_config = dict(
                explainerhub=dict(
                    **self._stored_params,
                    dashboards=[
                        dashboard.to_yaml(
                            return_dict=True,
                            explainerfile=dashboard.name + "_explainer.joblib",
                            dump_explainer=dump_explainers,
                        )
                        for dashboard in self.dashboards
                    ],
                )
            )
        else:
            for dashboard in self.dashboards:
                print(f"A guardar {dashboard.name}_dashboard.yaml...") # Traduzido
                dashboard.to_yaml(
                    filepath.parent / (dashboard.name + "_dashboard.yaml"),
                    explainerfile=filepath.parent
                    / (dashboard.name + f"_explainer.{pickle_type}"),
                    dump_explainer=dump_explainers,
                )
            hub_config = dict(
                explainerhub=dict(
                    **self._stored_params,
                    dashboards=[
                        dashboard.name + "_dashboard.yaml"
                        for dashboard in self.dashboards
                    ],
                )
            )

        if return_dict:
            return hub_config

        if filepath is None:
            return yaml.dump(hub_config)

        filepath = Path(filepath)
        print(f"A guardar {filepath}...") # Traduzido
        yaml.dump(hub_config, open(filepath, "w"))
        return

    def _store_params(self, no_store=None, no_attr=None, no_param=None):
        """Stores the parameter of the class to instance attributes and
        to a ._stored_params dict. You can optionall exclude all or some
        parameters from being stored.

        Args:
            no_store ({bool, List[str]}, optional): If True do not store any
                parameters to either attribute or _stored_params dict. If
                a list of str, then do not store parameters with those names.
                Defaults to None.
            no_attr ({bool, List[str]},, optional): . If True do not store any
                parameters to class attribute. If
                a list of str, then do not store parameters with those names.
                Defaults to None.
            no_param ({bool, List[str]},, optional): If True do not store any
                parameters to _stored_params dict. If
                a list of str, then do not store parameters with those names.
                Defaults to None.
        """
        if not hasattr(self, "_stored_params"):
            self._stored_params = {}

        frame = sys._getframe(1)
        args = frame.f_code.co_varnames[1 : frame.f_code.co_argcount]
        args_dict = {arg: frame.f_locals[arg] for arg in args}

        if "kwargs" in frame.f_locals:
            args_dict["kwargs"] = frame.f_locals["kwargs"]

        if isinstance(no_store, bool) and no_store:
            return
        else:
            if no_store is None:
                no_store = tuple()

        if isinstance(no_attr, bool) and no_attr:
            dont_attr = True
        else:
            if no_attr is None:
                no_attr = tuple()
            dont_attr = False

        if isinstance(no_param, bool) and no_param:
            dont_param = True
        else:
            if no_param is None:
                no_param = tuple()
            dont_param = False

        for name, value in args_dict.items():
            if not dont_attr and name not in no_store and name not in no_attr:
                setattr(self, name, value)
            if not dont_param and name not in no_store and name not in no_param:
                self._stored_params[name] = value

    def _instantiate_dashboards(self, dashboards, **kwargs):
        """Instantiate a list of dashboards and copy to logins to the ExplainerHub self.logins."""
        dashboard_list = []
        for i, dashboard in enumerate(dashboards):
            if dashboard.name is None:
                print(
                    "Lembrete, pode definir .name e .description do ExplainerDashboard " # Traduzido
                    "para controlar o caminho url do painel. Agora " # Traduzido
                    f"a usar por defeito name=dashboard{i+1} e descrição padrão...", # Traduzido
                    flush=True,
                )
                dashboard_name = f"dashboard{i+1}"
            else:
                dashboard_name = dashboard.name
            if dashboard_name in self.__reserved_names:
                raise ValueError(
                    f"ERRO! .name do painel não deve estar em {self.__reserved_names}, mas foi encontrado '{dashboard_name}'!" # Traduzido
                )
            update_params = dict(
                server=self.app,
                name=dashboard_name,
                url_base_pathname=f"/{self.base_route}/{dashboard_name}/",
                mode="dash",
            )
            if dashboard.logins is not None:
                for user, password in dashboard.logins:
                    if user not in self.logins:
                        self.add_user(user, password)
                    else:
                        print(
                            f"Aviso: {user} em {dashboard.name} já existe nos logins do " # Traduzido
                            "ExplainerHub! Não será adicionado aos logins..." # Traduzido
                        )
                    self.add_user_to_dashboard(dashboard_name, user)
            config = deepcopy(dashboard.to_yaml(return_dict=True))
            config["dashboard"]["params"]["logins"] = None

            dashboard_list.append(
                ExplainerDashboard.from_config(
                    dashboard.explainer,
                    config,
                    **update_kwargs(kwargs, **update_params),
                )
            )
        return dashboard_list

    @staticmethod
    def _validate_users_file(users_file: Path):
        """validat that user_json is a well formed .json file.
        If it does not exist, then create an empty .json file.
        """
        if users_file is not None:
            if not Path(users_file).exists():
                users_db = dict(users={}, dashboard_users={})
                if str(users_file).endswith(".json"):
                    json.dump(users_db, open(Path(users_file), "w"))
                elif str(users_file).endswith(".yaml"):
                    yaml.dump(users_db, open(Path(users_file), "w"))

            if str(users_file).endswith(".json"):
                users_db = json.load(open(Path(users_file)))
            elif str(users_file).endswith(".yaml"):
                users_db = yaml.safe_load(open(str(users_file), "r"))
            else:
                raise ValueError("users_file deve terminar em .json ou .yaml!") # Traduzido

            assert "users" in users_db, f"{users_file} deve conter um dict 'users'!" # Traduzido
            assert (
                "dashboard_users" in users_db
            ), f"{users_file} deve conter um dict 'dashboard_users'!" # Traduzido

    def _hash_logins(self, logins: List[List], add_to_users_file: bool = False):
        """Turn a list of [user, password] pairs into a Flask-Login style user
        dictionary with hashed passwords. If passwords already in hash-form
        then simply copy them.

        Args:
            logins (List[List]): List of logins e.g.
                logins = [['user1', 'password1], ['user2', 'password2]]
            add_to_users_file (bool, optional): Add the users to
                users.yaml. Defaults to False.

        Returns:
            dict
        """
        logins_dict = {}
        if logins is None:
            return logins_dict

        regex = re.compile(
            r"^pbkdf2:sha256:[0-9]+\$[a-zA-Z0-9]+\$[a-z0-9]{64}$", re.IGNORECASE
        )

        for username, password in logins:
            if re.search(regex, password) is not None:
                logins_dict[username] = dict(username=username, password=password)
                if add_to_users_file and self.users_file is not None:
                    self._add_user_to_file(
                        self.users_file, username, password, already_hashed=True
                    )
            else:
                logins_dict[username] = dict(
                    username=username,
                    password=generate_password_hash(password, method="pbkdf2:sha256"),
                )
                if add_to_users_file and self.users_jfile is not None:
                    self._add_user_to_file(self.users_file, username, password)
        return logins_dict

    @staticmethod
    def _load_users_db(users_file: Path):
        if str(users_file).endswith(".json"):
            users_db = json.load(open(Path(users_file)))
        elif str(users_file).endswith(".yaml"):
            users_db = yaml.safe_load(open(str(users_file), "r"))
        else:
            raise ValueError("users_file deve terminar em .json ou .yaml!") # Traduzido
        return users_db

    @staticmethod
    def _dump_users_db(users_db, users_file: Path):
        if str(users_file).endswith(".json"):
            json.dump(users_db, open(Path(users_file), "w"))
        elif str(users_file).endswith(".yaml"):
            yaml.dump(users_db, open(Path(users_file), "w"))
        else:
            raise ValueError("users_file deve terminar em .json ou .yaml!") # Traduzido

    def _dump_all_users_to_file(self, output_users_file: Path = None):
        """Stores all users (both on file and in the instance) to single users_file.
        Users in the instance overwrite users in users_file.

        Args:
            output_users_file (Path, optional): users_file to store users in.
                By default equal to self.users_file.
        """
        users_db = ExplainerHub._load_users_db(self.users_file)
        users_db["users"].update(self.logins)
        for db, instance_users in self.db_users.items():
            file_users = users_db["dashboard_users"].get(db) or []
            dashboard_users = sorted(list(set(file_users + instance_users)))
            users_db["dashboard_users"][db] = dashboard_users

        if output_users_file is None:
            output_users_file = self.users_file
        ExplainerHub._dump_users_db(users_db, output_users_file)

    @staticmethod
    def _add_user_to_file(
        users_file: Path, username: str, password: str, already_hashed=False
    ):
        """Add a user to a user_json .json file.

        Args:
            user_json (Path): json file, e.g 'users.json'
            username (str): username to add
            password (str): password to add
            already_hashed (bool, optional): If already hashed then do not hash
                the password again. Defaults to False.
        """
        users_db = ExplainerHub._load_users_db(users_file)
        users_db["users"][username] = dict(
            username=username,
            password=password
            if already_hashed
            else generate_password_hash(password, method="pbkdf2:sha256"),
        )
        ExplainerHub._dump_users_db(users_db, users_file)

    @staticmethod
    def _add_user_to_dashboard_file(users_file: Path, dashboard: str, username: str):
        """Add a user to dashboard_users inside a json file

        Args:
            user_json (Path): json file e.g. 'users.json'
            dashboard (str): name of dashboard
            username (str): username
        """
        users_db = ExplainerHub._load_users_db(users_file)

        dashboard_users = users_db["dashboard_users"].get(dashboard)
        if dashboard_users is None:
            dashboard_users = [username]
        else:
            dashboard_users = sorted(list(set(dashboard_users + [username])))
        users_db["dashboard_users"][dashboard] = dashboard_users

        ExplainerHub._dump_users_db(users_db, users_file)

    @staticmethod
    def _delete_user_from_file(users_file: Path, username: str):
        """delete user from user_json .json file.

        Also removes user from all dashboard_user lists.

        Args:
            user_json (Path): json file e.g. 'users.json'
            username (str): username to delete
        """
        users_db = ExplainerHub._load_users_db(users_file)
        try:
            del users_db["users"][username]
        except:
            pass
        for dashboard in users_db["dashboard_users"].keys():
            dashboard_users = users_db["dashboard_users"].get(dashboard)
            if dashboard_users is not None:
                dashboard_users = sorted(list(set(dashboard_users) - {username}))
                users_db["dashboard_users"][dashboard] = dashboard_users

        ExplainerHub._dump_users_db(users_db, users_file)

    @staticmethod
    def _delete_user_from_dashboard_file(
        users_file: Path, dashboard: str, username: str
    ):
        """remove a user from a specific dashboard_users list inside a users.json file

        Args:
            user_json (Path): json file, e.g. 'users.json'
            dashboard (str): name of the dashboard
            username (str): name of the user to remove
        """
        users_db = ExplainerHub._load_users_db(users_file)
        dashboard_users = users_db["dashboard_users"].get(dashboard)
        if dashboard_users is not None:
            dashboard_users = sorted(list(set(dashboard_users) - {username}))
            if not dashboard_users:
                del users_db["dashboard_users"][dashboard]
            else:
                users_db["dashboard_users"][dashboard] = dashboard_users
            ExplainerHub._dump_users_db(users_db, users_file)

    def add_user(self, username: str, password: str, add_to_users_file: bool = False):
        """add a user with username and password.

        Args:
            username (str): username
            password (str): password
            add_to_users_file(bool, optional): Add the user to the .yaml file defined
                in self.users_file instead of to self.logins. Defaults to False.
        """
        if add_to_users_file and self.users_file is not None:
            self._add_user_to_file(self.users_file, username, password)
        else:
            self.logins[username] = dict(
                username=username,
                password=generate_password_hash(password, method="pbkdf2:sha256"),
            )

    def add_user_to_dashboard(
        self, dashboard: str, username: str, add_to_users_file: bool = False
    ):
        """add a user to a specific dashboard. If

        Args:
            dashboard (str): name of dashboard
            username (str): user to add to dashboard
            add_to_users_file (bool, optional): add the user to the .yaml or .json file defined
                in self.users_file instead of to self.db_users. Defaults to False.
        """
        if add_to_users_file and self.users_file is not None:
            self._add_user_to_dashboard_file(self.users_file, dashboard, username)
        else:
            dashboard_users = self.db_users.get(dashboard)
            dashboard_users = dashboard_users if dashboard_users is not None else []
            dashboard_users = sorted(list(set(dashboard_users + [username])))
            self.db_users[dashboard] = dashboard_users

    @property
    def users(self):
        """returns a list of all users, both in users.json and in self.logins"""
        users = []
        if self.users_file is not None and Path(self.users_file).exists():
            users = list(self._load_users_db(self.users_file)["users"].keys())
        if self.logins is not None:
            users.extend(list(self.logins.keys()))
        return users

    @property
    def dashboards_with_users(self):
        """returns a list of all dashboards that have a restricted list of users
        that can access it"""
        dashboards = []
        if self.users_file is not None and Path(self.users_file).exists():
            dashboards = list(
                self._load_users_db(self.users_file)["dashboard_users"].keys()
            )
        if self.logins is not None:
            dashboards.extend(list(self.db_users.keys()))
        return dashboards

    @property
    def dashboard_users(self):
        """return a dict with the list of users per dashboard"""
        dashboard_users = {}
        if self.users_file is not None and Path(self.users_file).exists():
            dashboard_users.update(
                self._load_users_db(self.users_file)["dashboard_users"]
            )
        if self.db_users is not None:
            for dashboard, users in self.db_users.items():
                if not dashboard in dashboard_users:
                    dashboard_users[dashboard] = users
                else:
                    dashboard_users[dashboard] = sorted(
                        list(set(dashboard_users[dashboard] + users))
                    )
        return dashboard_users

    def get_dashboard_users(self, dashboard: str):
        """return all users that have been approved to use a specific dashboard

        Args:
            dashboard (str): dashboard

        Returns:
            List
        """
        dashboard_users = []
        if self.users_file is not None:
            file_users = self._load_users_db(self.users_file)["dashboard_users"].get(
                dashboard
            )
            if file_users is not None:
                dashboard_users = file_users
        if self.db_users is not None:
            param_users = self.db_users.get(dashboard)
            if param_users is not None:
                dashboard_users.extend(param_users)
        dashboard_users = list(set(dashboard_users))
        return dashboard_users

    def _validate_user(self, user):
        """validation function for SimpleLogin. Returns True when user should
        be given access (i.e. no users defined or password correct) and False
        when user should be rejected.

        Args:
            user (dict(username, password)): dictionary with a username and
                password key.

        Returns:
            bool
        """
        if not self.users:
            return True
        users_db = (
            self._load_users_db(self.users_file)["users"]
            if self.users_file is not None
            else {}
        )
        if not self.logins.get(user["username"]) and not users_db.get(user["username"]):
            return False
        if user["username"] in users_db:
            stored_password = users_db[user["username"]]["password"]
        else:
            stored_password = self.logins[user["username"]]["password"]
        if check_password_hash(stored_password, user["password"]):
            return True
        return False

    @staticmethod
    def _protect_dashviews(dashapp: dash.Dash, username: List[str] = None):
        """Wraps a dash dashboard inside a login_required decorator to make sure
        unauthorized viewers cannot access it.

        Args:
            dashapp (dash.Dash): a dash app
            username (List[str], optional): list of usernames that can access
                this specific dashboard. Defaults to None (all registered users
                can access)
        """
        for view_func in dashapp.server.view_functions:
            if view_func.startswith(dashapp.config.url_base_pathname):
                dashapp.server.view_functions[view_func] = login_required(
                    username=username
                )(dashapp.server.view_functions[view_func])

    def _get_index_page(self):
        """Returns the front end of ExplainerHub:

        - title
        - description
        - links and description for each dashboard

        Returns:
            dbc.Container
        """

        def dashboard_decks(dashboards, n_cols):
            full_rows = int(len(dashboards) / n_cols)
            n_last_row = len(dashboards) % n_cols
            card_decks = []
            for i in range(0, full_rows * n_cols, n_cols):
                card_decks.append(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    [
                                        html.H3(
                                            dashboard.title, className="card-title" # Assume dashboard.title traduzido
                                        ),
                                    ]
                                ),
                                dbc.CardBody(
                                    [
                                        html.H6(dashboard.description), # Assume dashboard.description traduzido
                                    ]
                                ),
                                dbc.CardFooter(
                                    [
                                        dbc.CardLink(
                                            "Ir para o painel", # Traduzido
                                            href=f"/{self.base_route}/{dashboard.name}",
                                            external_link=True,
                                        ),
                                    ]
                                ),
                            ],
                            class_name="h-100",
                        )
                        for dashboard in dashboards[i : i + n_cols]
                    ]
                )
            if n_last_row > 0:
                last_row = [
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                [
                                    html.H3(dashboard.title, className="card-title"), # Assume dashboard.title traduzido
                                ]
                            ),
                            dbc.CardBody(
                                [
                                    html.H6(dashboard.description), # Assume dashboard.description traduzido
                                ]
                            ),
                            dbc.CardFooter(
                                [
                                    dbc.CardLink(
                                        "Ir para o painel", # Traduzido
                                        href=f"/{self.base_route}/{dashboard.name}",
                                        external_link=True,
                                    ),
                                ]
                            ),
                        ],
                        class_name="h-100",
                    )
                    for dashboard in dashboards[
                        full_rows * n_cols : full_rows * n_cols + n_last_row
                    ]
                ]
                for i in range(len(last_row), n_cols):
                    last_row.append(dbc.Card([], style=dict(border="none")))
                card_decks.append(last_row)
            return card_decks

        header = html.Div(
            [
                dbc.Container(
                    [
                        html.H1(self.title, className="display-3"), # Assume self.title traduzido
                        html.Hr(className="my-2"),
                        html.P(self.description, className="lead"), # Assume self.description traduzido
                    ],
                    fluid=True,
                    class_name="py-3",
                )
            ],
            className="p-3 bg-light rounded-3",
            style=dict(marginTop=40),
        )

        if self.masonry:
            dashboard_rows = [
                dbc.Row([dbc.Col([dbc.CardColumns(dashboard_cards(self.dashboards))])])
            ]
        else:
            dashboard_rows = [
                dbc.Row([dbc.Col(card) for card in deck], class_name="mt-4 g-4")
                for deck in dashboard_decks(self.dashboards, self.n_dashboard_cols)
            ]

        if hasattr(self, "index_page"):
            index_page = self.index_page
        else:
            index_page = dash.Dash(
                __name__,
                server=self.app,
                url_base_pathname=f"{self.hub_base_url}",
                external_stylesheets=[self.bootstrap]
                if self.bootstrap is not None
                else None,
            )
            index_page.title = self.title # Assume self.title traduzido

        index_page.layout = dbc.Container(
            [
                dbc.Row([dbc.Col([header])]),
                dbc.Row([dbc.Col([html.H2("Painéis:")])]), # Traduzido
                *dashboard_rows,
            ],
            fluid=self.fluid,
        )
        return index_page

    def to_html(self):
        """
        returns static html version of the hub landing page
        """

        def dashboard_cards(dashboards, n_cols):
            full_rows = int(len(dashboards) / n_cols)
            n_last_row = len(dashboards) % n_cols
            card_decks = []
            for i in range(0, full_rows * n_cols, n_cols):
                card_decks.append(
                    [
                        to_html.dashboard_card(
                            dashboard.title, # Assume traduzido
                            dashboard.description, # Assume traduzido
                            dashboard.name + ".html",
                        )
                        for dashboard in dashboards[i : i + n_cols]
                    ]
                )
            if n_last_row > 0:
                last_row = [
                    to_html.dashboard_card(
                        dashboard.title, dashboard.description, dashboard.name + ".html" # Assume traduzido
                    )
                    for dashboard in dashboards[
                        full_rows * n_cols : full_rows * n_cols + n_last_row
                    ]
                ]
                for i in range(len(last_row), n_cols):
                    last_row.append(to_html.card("", border=False))
                card_decks.append(last_row)
            return card_decks

        html = to_html.jumbotron(self.title, self.description) # Assume traduzido
        html += to_html.card_rows(
            *dashboard_cards(self.dashboards, self.n_dashboard_cols)
        )
        return self._hub_page(html, static=True)

    def save_html(
        self, filename: Union[str, Path] = None, save_dashboards: bool = True
    ):
        """Store output of to_html to a file

        Args:
            filename (str, Path): filename to store html
            save_dashboard (bool): save dashboards the make up the hub into
                individual static html files.
        """
        html = self.to_html()
        if filename is None:
            return html
        with open(filename, "w") as f:
            print(f"A guardar hub para {filename}...") # Traduzido
            f.write(html)
        if save_dashboards:
            for db in self.dashboards:
                print(f"A guardar painel {db.name} para {db.name}.html...") # Traduzido
                db.save_html(db.name + ".html")

    def to_zip(self, filename: Union[str, Path], name: str = "explainerhub"):
        """Store static version of ExplainerHub to a zipfile along with static
        versions of all underlying dashboards.

        Args:
            filename (Union[str, Path], optional): filename of zip file, eg. "hub.zip".
            name (str): name for the directory inside the zipfile
        """
        import zipfile

        zf = zipfile.ZipFile(Path(filename), "w")
        zf.writestr(f"/{name}/index.html", self.to_html())
        for db in self.dashboards:
            zf.writestr(f"/{name}/" + db.name + ".html", db.to_html())
        zf.close()
        print(f"Guardada versão html estática do ExplainerHub em {filename}...") # Traduzido

    def _hub_page(self, route, static=False):
        """Returns a html bootstrap wrapper around a particular flask route (hosting an ExplainerDashbaord)
        It contains:
        - a NavBar with links to all dashboards
        - an iframe containing the flask route
        """
        if static:
            page = """
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
            """
            dbs = [(f"{db.name}.html", db.title) for db in self.dashboards] # Assume db.title traduzido
        else:
            page = f"""
            <script type="text/javascript" src=f"{self.app.static_url_path}/jquery-3.5.1.slim.min.js"></script>
            <script type="text/javascript" src=f"{self.app.static_url_path}/bootstrap.min.js"></script>
            <link type="text/css" rel="stylesheet" href="{f'{self.app.static_url_path}/bootstrap.min.css' if self.bootstrap is None else self.bootstrap}"/>
            <link rel="shortcut icon" href=f"{self.app.static_url_path}/favicon.ico">
            """
            dbs = [
                (f"/{self.base_route}/_{db.name}", db.title) for db in self.dashboards # Assume db.title traduzido
            ]

        page_login_required = self.users and not self.dbs_open_by_default

        page += f"""
        <title>{self.title}</title>  <!-- Assume self.title traduzido -->
        <body>
            <div class="container{'-fluid' if self.fluid else ''} px-4">
                <nav class="navbar navbar-expand navbar-light bg-light">
                    <div class="container-fluid">
                        <a href="{self.index_route if not static else '#'}" class="navbar-brand">
                        <h1>{self.title}</h1> <!-- Assume self.title traduzido -->
                        </a>
                        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                            <span class="navbar-toggler-icon"></span>
                        </button>
                        <form class="d-flex">
                            <div class="collapse navbar-collapse" id="navbarSupportedContent">
                                <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                                    <li class="nav-item dropdown">
                                    <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" role="button" data-bs-toggle="dropdown" aria-expanded="false">
                                        Painéis <!-- Traduzido -->
                                    </a>
                                    <ul class="dropdown-menu" aria-labelledby="navbarDropdown">
                                        {"".join([f'<li><a n_clicks_timestamp="-1" data-rr-ui-dropdown-item="" class="dropdown-item" href="{url}">{name}</a></li>' for url, name in dbs])} <!-- Assume name (db.title) traduzido -->
                                    </ul>
                                    </li>
                                    {'<li class="nav-item"><a class="nav-link" href="/logout">Sair</a></li>' if not static and page_login_required else ''} <!-- Traduzido -->
                                </ul>
                            </div>
                        </form>
                    </div>
                </nav>
        """
        if static:
            page += f"\n<div>\n{route}\n</div>\n"
        else:
            page += f"""
            
            <div class="mt-4 embed-responsive" style="min-height: {self.min_height}px">
                 <iframe 
                         src="{route}"
                         style="overflow-x: hidden; overflow-y: visible; position: absolute; width: 95%; height: 100%; background: transparent"
                ></iframe>
            </div>
            """
        page += """</div>
        </body>
        """
        return page

    def _add_flask_routes(self, app):
        """adds the index route "/" with the index_page
        and routes for each dashboard with a navbar and an iframe, e.g. "/_dashboard1"

        If you pass no_index to the contructor, this method does not get called.

        Args:
            app (flask.Flask): flask app to add routes to.
        """
        if self.users and not self.dbs_open_by_default:

            @app.route(f"{self.index_route}")
            @login_required
            def index_route():
                return self._hub_page(f"{self.hub_base_url}")

            def dashboard_route(dashboard):
                def inner():
                    return self._hub_page(f"/{self.base_route}/{dashboard.name}/")

                inner.__name__ = "return_dashboard_" + dashboard.name
                return inner

            for dashboard in self.dashboards:
                app.route(f"/{self.base_route}/_{dashboard.name}")(
                    login_required(dashboard_route(dashboard))
                )
        else:

            @app.route(f"{self.index_route}")
            def index_route():
                return self._hub_page(f"{self.hub_base_url}")

            def dashboard_route(dashboard):
                def inner():
                    return self._hub_page(f"/{self.base_route}/{dashboard.name}/")

                inner.__name__ = "return_dashboard_" + dashboard.name
                return inner

            for dashboard in self.dashboards:
                app.route(f"/{self.base_route}/_{dashboard.name}")(
                    dashboard_route(dashboard)
                )

        if self.add_dashboard_route:
            add_dashboard_pattern = re.compile(r"(\/add_dashboard\/)(.+)")
            remove_dashboard_pattern = re.compile(r"(\/remove_dashboard\/)(.+)")

            @self.app.before_request
            def add_dashboard():
                """
                Try to generate a new dashboard when user accesses /add_dashboard/dashboard_name.

                Looks up if dashboards/dashboard_name/dashboard.yaml exists, and if so add a dashboard
                with dashboard_name to the hub.
                """
                add_dashboard_match = add_dashboard_pattern.match(request.path)
                if add_dashboard_match:
                    try:
                        _, dashboard_path = add_dashboard_match.groups()
                        if self.add_dashboard_pattern is not None:
                            dashboard_path = self.add_dashboard_pattern.format(
                                dashboard_path
                            )
                        if (
                            dashboard_path.endswith(".yaml")
                            and Path(dashboard_path).exists()
                        ):
                            db = ExplainerDashboard.from_config(dashboard_path)
                            dashboard_name = self.add_dashboard(
                                db,
                                bootstrap=f"{self.app.static_url_path}/bootstrap.min.css",
                            )
                            return redirect(f"/dashboards/_{dashboard_name}", code=302)
                    except:
                        print("ERRO: Falha ao adicionar painel!", flush=True) # Traduzido
                    return redirect("/", code=302)

                remove_dashboard_match = remove_dashboard_pattern.match(request.path)
                if remove_dashboard_match:
                    try:
                        _, dashboard_name = remove_dashboard_match.groups()
                        if dashboard_name in self.dashboard_names:
                            self.remove_dashboard(dashboard_name)
                    except:
                        print("ERRO: Falha ao remover painel!", flush=True) # Traduzido
                    return redirect(f"/", code=302)

    def flask_server(self):
        """return the Flask server inside the class instance"""
        return self.app

    def run(self, port=None, host="0.0.0.0", use_waitress=False, **kwargs):
        """start the ExplainerHub.

        Args:
            port (int, optional): Override default port. Defaults to None.
            host (str, optional): host name to run dashboard. Defaults to '0.0.0.0'.
            use_waitress (bool, optional): Use the waitress python web server
                instead of the Flask development server. Defaults to False.
            **kwargs: will be passed forward to either waitress.serve() or app.run()
        """
        if port is None:
            port = self.port
        print(
            f"A iniciar ExplainerHub em http://{host}:{port}{self.index_route}", # Traduzido
            flush=True,
        )
        if use_waitress:
            import waitress

            waitress.serve(self.app, host=host, port=port, **kwargs)
        else:
            self.app.run(host=host, port=port, **kwargs)


class InlineExplainer:
    """
    Run a single tab inline in a Jupyter notebook using specific method calls.
    """

    def __init__(
        self,
        explainer: BaseExplainer,
        mode: str = "inline",
        width: int = 1000,
        height: int = 800,
        port: int = 8050,
        **kwargs,
        ):
        """
        :param explainer: an Explainer object
        :param mode: either 'inline', 'jupyterlab' or 'external'
        :type mode: str, optional
        :param width: width in pixels of inline iframe
        :param height: height in pixels of inline iframe
        :param port: port to run if mode='external'
        """
        assert mode in [
            "inline",
            "external",
            "jupyterlab",
            ], "mode deve ser 'inline', 'external' ou 'jupyterlab'!" # Traduzido
        self._explainer = explainer
        self._mode = mode
        self._width = width
        self._height = height
        self._port = port
        self._kwargs = kwargs
        self.tab = InlineExplainerTabs(self, "tabs")
        """subclass with InlineExplainerTabs layouts, e.g. InlineExplainer(explainer).tab.modelsummary()"""
        self.shap = InlineShapExplainer(self, "shap")
        """subclass with InlineShapExplainer layouts, e.g. InlineExplainer(explainer).shap.dependence()"""
        self.classifier = InlineClassifierExplainer(self, "classifier")
        """subclass with InlineClassifierExplainer plots, e.g. InlineExplainer(explainer).classifier.confusion_matrix()"""
        self.regression = InlineRegressionExplainer(self, "regression")
        """subclass with InlineRegressionExplainer plots, e.g. InlineExplainer(explainer).regression.residuals()"""
        self.decisiontrees = InlineDecisionTreesExplainer(self, "decisiontrees")
        """subclass with InlineDecisionTreesExplainer plots, e.g. InlineExplainer(explainer).decisiontrees.decisiontrees()"""

    def terminate(self, port=None, token=None):
        """terminate an InlineExplainer on particular port.

        You can kill any JupyterDash dashboard from any ExplainerDashboard
        by specifying the right port.

        Args:
            port (int, optional): port on which the InlineExplainer is running.
                        Defaults to the last port the instance had started on.
            token (str, optional): JupyterDash._token class property.
                Defaults to the _token of the JupyterDash in the current namespace.

        Raises:
            ValueError: if can't find the port to terminate.
        """
        if port is None:
            port = self._port
        if token is None:
            token = JupyterDash._token

        shutdown_url = f"http://localhost:{port}/_shutdown_{token}"
        print(f"A tentar encerrar o painel na porta {port}...") # Traduzido
        try:
            response = requests.get(shutdown_url)
        except Exception as e:
            print(f"Algo parece ter falhado: {e}") # Traduzido

    def _run_app(self, app, **kwargs):
        """Starts the dashboard either inline or in a seperate tab

        :param app: the JupyterDash app to be run
        :type mode: JupyterDash app instance
        """
        pio.templates.default = "none"
        if self._mode in ["inline", "jupyterlab"]:
            app.run_server(
                mode=self._mode, width=self._width, height=self._height, port=self._port
            )
        elif self._mode == "external":
            app.run_server(mode=self._mode, port=self._port, **self._kwargs)
        else:
            raise ValueError(
                "mode deve ser 'inline', 'jupyterlab' ou 'external'!" # Traduzido
            )

    def _run_component(self, component, title):
        app = JupyterDash(__name__)
        app.title = title # Assume title traduzido
        app.layout = component.layout()
        component.register_callbacks(app)
        self._run_app(app)

    @delegates_kwargs(ImportancesComponent)
    @delegates_doc(ImportancesComponent)
    def importances(self, title="Importâncias", **kwargs): # Traduzido title
        """Runs model_summary tab inline in notebook"""
        comp = ImportancesComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    def model_stats(self, title="Estatísticas do Modelo", **kwargs): # Traduzido title
        """Runs model_stats inline in notebook"""
        if self._explainer.is_classifier:
            comp = ClassifierModelStatsComposite(self._explainer, **kwargs)
        elif self._explainer.is_regression:
            comp = RegressionModelStatsComposite(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PredictionSummaryComponent)
    @delegates_doc(PredictionSummaryComponent)
    def prediction(self, title="Previsão", **kwargs): # Traduzido title
        """Show contributions (permutation or shap) inline in notebook"""
        comp = PredictionSummaryComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    def random_index(self, title="Índice Aleatório", **kwargs): # Traduzido title
        """show random index selector inline in notebook"""
        if self._explainer.is_classifier:
            comp = ClassifierRandomIndexComponent(self._explainer, **kwargs)
        elif self._explainer.is_regression:
            comp = RegressionRandomIndexComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PdpComponent)
    @delegates_doc(PdpComponent)
    def pdp(self, title="Gráficos de Dependência Parcial", **kwargs): # Traduzido title
        """Show contributions (permutation or shap) inline in notebook"""
        comp = PdpComponent(self._explainer, **kwargs)
        self._run_component(comp, title)


class InlineExplainerComponent:
    def __init__(self, inline_explainer, name):
        self._inline_explainer = inline_explainer
        self._explainer = inline_explainer._explainer
        self._name = name

    def _run_component(self, component, title):
        self._inline_explainer._run_component(component, title)

    def __repr__(self):
        component_methods = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and not method_name.startswith("_")
        ]

        return f"InlineExplainer.{self._name} tem os seguintes componentes: {', '.join(component_methods)}" # Traduzido


class InlineExplainerTabs(InlineExplainerComponent):
    @delegates_kwargs(ImportancesComposite)
    @delegates_doc(ImportancesComposite)
    def importances(self, title="Importâncias", **kwargs): # Traduzido title
        """Show contributions (permutation or shap) inline in notebook"""
        tab = ImportancesComposite(self._explainer, **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(RegressionModelStatsComposite)
    @delegates_doc(RegressionModelStatsComposite)
    def modelsummary(self, title="Resumo do Modelo", **kwargs): # Traduzido title
        """Runs model_summary tab inline in notebook"""
        if self._explainer.is_classifier:
            tab = ClassifierModelStatsComposite(self._explainer, **kwargs)
        else:
            tab = RegressionModelStatsComposite(self._explainer, **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(IndividualPredictionsComposite)
    @delegates_doc(IndividualPredictionsComposite)
    def contributions(self, title="Contribuições", **kwargs): # Traduzido title
        """Show contributions (permutation or shap) inline in notebook"""
        tab = IndividualPredictionsComposite(self._explainer, **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(WhatIfComposite)
    @delegates_doc(WhatIfComposite)
    def whatif(self, title="E Se...", **kwargs): # Traduzido title
        """Show What if... tab inline in notebook"""
        tab = WhatIfComposite(self._explainer, **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(ShapDependenceComposite)
    @delegates_doc(ShapDependenceComposite)
    def dependence(self, title="Dependência SHAP", **kwargs): # Traduzido title (já estava ok)
        """Runs shap_dependence tab inline in notebook"""
        tab = ShapDependenceComposite(self._explainer, **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(ShapInteractionsComposite)
    @delegates_doc(ShapInteractionsComposite)
    def interactions(self, title="Interações SHAP", **kwargs): # Traduzido title
        """Runs shap_interactions tab inline in notebook"""
        tab = ShapInteractionsComposite(self._explainer, **kwargs)
        self._run_component(tab, title)

    @delegates_kwargs(DecisionTreesComposite)
    @delegates_doc(DecisionTreesComposite)
    def decisiontrees(self, title="Árvores de Decisão", **kwargs): # Traduzido title
        """Runs shap_interactions tab inline in notebook"""
        tab = DecisionTreesComposite(self._explainer, **kwargs)
        self._run_component(tab, title)


class InlineShapExplainer(InlineExplainerComponent):
    @delegates_kwargs(ShapDependenceComposite)
    @delegates_doc(ShapDependenceComposite)
    def overview(self, title="Visão Geral SHAP", **kwargs): # Traduzido title
        """Runs shap_dependence tab inline in notebook"""
        comp = ShapDependenceComposite(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapSummaryComponent)
    @delegates_doc(ShapSummaryComponent)
    def summary(self, title="Resumo SHAP", **kwargs): # Traduzido title
        """Show shap summary inline in notebook"""
        comp = ShapSummaryComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapDependenceComponent)
    @delegates_doc(ShapDependenceComponent)
    def dependence(self, title="Dependência SHAP", **kwargs): # Traduzido title (já estava ok)
        """Show shap summary inline in notebook"""

        comp = ShapDependenceComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapInteractionsComposite)
    @delegates_doc(ShapInteractionsComposite)
    def interaction_overview(self, title="Visão Geral das Interações", **kwargs): # Traduzido title
        """Runs shap_dependence tab inline in notebook"""
        comp = ShapInteractionsComposite(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(InteractionSummaryComponent)
    @delegates_doc(InteractionSummaryComponent)
    def interaction_summary(self, title="Resumo das Interações SHAP", **kwargs): # Traduzido title
        """show shap interaction summary inline in notebook"""
        comp = InteractionSummaryComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(InteractionDependenceComponent)
    @delegates_doc(InteractionDependenceComponent)
    def interaction_dependence(self, title="Dependência das Interações SHAP", **kwargs): # Traduzido title
        """show shap interaction dependence inline in notebook"""
        comp = InteractionDependenceComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapContributionsGraphComponent)
    @delegates_doc(ShapContributionsGraphComponent)
    def contributions_graph(self, title="Gráfico de Contribuições", **kwargs): # Traduzido title
        """Show contributions (permutation or shap) inline in notebook"""
        comp = ShapContributionsGraphComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ShapContributionsTableComponent)
    @delegates_doc(ShapContributionsTableComponent)
    def contributions_table(self, title="Tabela de Contribuições", **kwargs): # Traduzido title
        """Show contributions (permutation or shap) inline in notebook"""
        comp = ShapContributionsTableComponent(self._explainer, **kwargs)
        self._run_component(comp, title)


class InlineClassifierExplainer(InlineExplainerComponent):
    @delegates_kwargs(ClassifierModelStatsComposite)
    @delegates_doc(ClassifierModelStatsComposite)
    def model_stats(self, title="Estatísticas do Modelo", **kwargs): # Traduzido title
        """Runs model_stats inline in notebook"""
        comp = ClassifierModelStatsComposite(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PrecisionComponent)
    @delegates_doc(PrecisionComponent)
    def precision(self, title="Gráfico de Precisão", **kwargs): # Traduzido title
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = PrecisionComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(CumulativePrecisionComponent)
    @delegates_doc(CumulativePrecisionComponent)
    def cumulative_precision(self, title="Gráfico de Precisão Acumulada", **kwargs): # Traduzido title
        """shows cumulative precision plot"""
        assert self._explainer.is_classifier
        comp = CumulativePrecisionComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ConfusionMatrixComponent)
    @delegates_doc(ConfusionMatrixComponent)
    def confusion_matrix(self, title="Matriz de Confusão", **kwargs): # Traduzido title
        """shows precision plot"""
        comp = ConfusionMatrixComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(LiftCurveComponent)
    @delegates_doc(LiftCurveComponent)
    def lift_curve(self, title="Curva Lift", **kwargs): # Traduzido title
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = LiftCurveComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ClassificationComponent)
    @delegates_doc(ClassificationComponent)
    def classification(self, title="Classificação", **kwargs): # Traduzido title
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = ClassificationComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(RocAucComponent)
    @delegates_doc(RocAucComponent)
    def roc_auc(self, title="Curva ROC AUC", **kwargs): # Traduzido title
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = RocAucComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PrAucComponent)
    @delegates_doc(PrAucComponent)
    def pr_auc(self, title="Curva PR AUC", **kwargs): # Traduzido title
        """shows precision plot"""
        assert self._explainer.is_classifier
        comp = PrAucComponent(self._explainer, **kwargs)
        self._run_component(comp, title)


class InlineRegressionExplainer(InlineExplainerComponent):
    @delegates_kwargs(RegressionModelStatsComposite)
    @delegates_doc(RegressionModelStatsComposite)
    def model_stats(self, title="Estatísticas do Modelo", **kwargs): # Traduzido title
        """Runs model_stats inline in notebook"""
        comp = RegressionModelStatsComposite(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(PredictedVsActualComponent)
    @delegates_doc(PredictedVsActualComponent)
    def pred_vs_actual(self, title="Previsto vs Real", **kwargs): # Traduzido title
        "shows predicted vs actual for regression"
        assert self._explainer.is_regression
        comp = PredictedVsActualComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(ResidualsComponent)
    @delegates_doc(ResidualsComponent)
    def residuals(self, title="Resíduos", **kwargs): # Traduzido title
        "shows residuals for regression"
        assert self._explainer.is_regression
        comp = ResidualsComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(RegressionVsColComponent)
    @delegates_doc(RegressionVsColComponent)
    def plots_vs_col(self, title="Gráficos vs Coluna", **kwargs): # Traduzido title
        "shows plots vs col for regression"
        assert self._explainer.is_regression
        comp = RegressionVsColComponent(self._explainer, **kwargs)
        self._run_component(comp, title)


class InlineDecisionTreesExplainer(InlineExplainerComponent):
    @delegates_kwargs(DecisionTreesComposite)
    @delegates_doc(DecisionTreesComposite)
    def overview(self, title="Visão Geral das Árvores de Decisão", **kwargs): # Traduzido title
        """shap decision tree composite inline in notebook"""
        comp = DecisionTreesComposite(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(DecisionTreesComponent)
    @delegates_doc(DecisionTreesComponent)
    def decisiontrees(self, title="Árvores de Decisão", **kwargs): # Traduzido title
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionTreesComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(DecisionPathTableComponent)
    @delegates_doc(DecisionPathTableComponent)
    def decisionpath_table(self, title="Tabela do Caminho de Decisão", **kwargs): # Traduzido title
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionPathTableComponent(self._explainer, **kwargs)
        self._run_component(comp, title)

    @delegates_kwargs(DecisionPathTableComponent) # Corrigido para Graph?
    @delegates_doc(DecisionPathTableComponent) # Corrigido para Graph?
    def decisionpath_graph(self, title="Gráfico do Caminho de Decisão", **kwargs): # Traduzido title
        """Runs decision_trees tab inline in notebook"""
        comp = DecisionPathGraphComponent(self._explainer, **kwargs) # Assumindo que existe DecisionPathGraphComponent
        self._run_component(comp, title)