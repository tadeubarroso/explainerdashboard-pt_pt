__all__ = [
    "DecisionTreesComponent",
    "DecisionPathTableComponent",
    "DecisionPathGraphComponent",
]

import dash
from dash import html, dcc, Input, Output, State, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..explainers import RandomForestExplainer, XGBExplainer
from ..dashboard_methods import *
from .classifier_components import ClassifierRandomIndexComponent
from .connectors import IndexConnector, HighlightConnector
from .. import to_html


class DecisionTreesComponent(ExplainerComponent):
    _state_props = dict(
        index=("decisiontrees-index-", "value"),
        highlight=("decisiontrees-highlight-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Decision Trees",
        name=None,
        subtitle="Displaying individual decision trees",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_highlight=False,
        hide_selector=False,
        hide_popout=False,
        index_dropdown=True,
        pos_label=None,
        index=None,
        highlight=None,
        higher_is_better=True,
        description=None,
        **kwargs,
    ):
        """Show prediction from individual decision trees inside RandomForest component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Decision Trees".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title, Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector. Defaults to False.
            hide_highlight (bool, optional): Hide tree highlight selector. Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            hide_popout (bool, optional): hide popout button
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display. Defaults to None.
            highlight (int, optional): Initial tree to highlight. Defaults to None.
            higher_is_better (bool, optional): up is green, down is red. If False
                flip the colors. (for gbm models only)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "decisiontrees-index-" + self.name
        self.highlight_name = "decisiontrees-highlight-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "decisiontrees-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if isinstance(self.explainer, RandomForestExplainer):
            if self.description is None:
                self.description = """
            Mostrar a previsão de cada árvore individual numa random forest.
            Isto demonstra como uma random forest é simplesmente uma média de um
            conjunto (ensemble) de árvores de decisão.
            """
            if self.subtitle == "Exibindo árvores de decisão individuais":
                self.subtitle += " dentro da Random Forest"
        elif isinstance(self.explainer, XGBExplainer):
            if self.description is None:
                self.description = """
            Mostra as contribuições marginais de cada árvore de decisão num
            conjunto (ensemble) xgboost para a previsão final. Isto demonstra que
            um modelo xgboost é simplesmente uma soma de árvores de decisão individuais.
            """
            if self.subtitle == "Exibindo árvores de decisão individuais":
                self.subtitle += " dentro do modelo xgboost"
        else:
            if self.description is None:
                self.description = ""

        self.popout = GraphPopout(
            "decisiontrees-" + self.name + "popout",
            "decisiontrees-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies("preds", "pred_probas")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="decisiontrees-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="decisiontrees-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="decisiontrees-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar {self.explainer.index_name} para exibir árvores de decisão",
                                                target="decisiontrees-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Destacar árvore:",
                                                id="decisiontrees-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar uma árvore específica para destacar. Também pode "
                                                "destacar clicando numa barra específica no gráfico de barras.",
                                                target="decisiontrees-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="decisiontrees-highlight-"
                                                + self.name,
                                                options=[
                                                    {"label": str(tree), "value": tree}
                                                    for tree in range(
                                                        self.explainer.no_of_trees
                                                    )
                                                ],
                                                value=self.highlight,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_highlight,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="decisiontrees-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
                                            figure={},
                                        ),
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [self.popout.layout()], md=2, align="start"
                                    ),
                                    hide=self.hide_popout,
                                ),
                            ],
                            justify="end",
                        ),
                    ]
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        args["highlight"] = (
            None if args["highlight"] is None else int(args["highlight"])
        )
        if args["index"] is not None:
            fig = self.explainer.plot_trees(
                args["index"],
                highlight_tree=args["highlight"],
                pos_label=args["pos_label"],
                higher_is_better=self.higher_is_better,
            )
            html = to_html.fig(fig)
        else:
            html = "no index selected"
        html = to_html.card(html, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("decisiontrees-graph-" + self.name, "figure"),
            [
                Input("decisiontrees-index-" + self.name, "value"),
                Input("decisiontrees-highlight-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_tree_graph(index, highlight, pos_label):
            if index is None or not self.explainer.index_exists(index):
                raise PreventUpdate
            highlight = None if highlight is None else int(highlight)
            return self.explainer.plot_trees(
                index,
                highlight_tree=highlight,
                pos_label=pos_label,
                higher_is_better=self.higher_is_better,
            )

        @app.callback(
            Output("decisiontrees-highlight-" + self.name, "value"),
            [Input("decisiontrees-graph-" + self.name, "clickData")],
        )
        def update_highlight(clickdata):
            highlight_tree = (
                int(clickdata["points"][0]["text"].split("tree no ")[1].split(":")[0])
                if clickdata is not None
                else None
            )
            if highlight_tree is not None:
                return highlight_tree
            raise PreventUpdate


class DecisionPathTableComponent(ExplainerComponent):
    _state_props = dict(
        index=("decisionpath-table-index-", "value"),
        highlight=("decisionpath-table-highlight-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Tabela do Caminho de Decisão",
        name=None,
        subtitle="Caminho de decisão através da árvore de decisão",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_highlight=False,
        hide_selector=False,
        index_dropdown=True,
        pos_label=None,
        index=None,
        highlight=None,
        description=None,
        **kwargs,
    ):
        """Display a table of the decision path through a particular decision tree

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Decision path table".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title, Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector.
                        Defaults to False.
            hide_highlight (bool, optional): Hide tree index selector.
                        Defaults to False.
            hide_selector (bool, optional): hide pos label selectors.
                        Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                        of free text input. Defaults to True.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display decision
                        path for. Defaults to None.
            highlight (int, optional): Initial tree idx to display decision
                        path for. Defaults to None.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "decisionpath-table-index-" + self.name
        self.highlight_name = "decisionpath-table-highlight-" + self.name

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "decisionpath-table-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if self.description is None:
            self.description = """
        Mostra o caminho que uma observação percorreu numa árvore de decisão específica.
        """
        self.register_dependencies("shadow_trees")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="decisionpath-table-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="decisionpath-table-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="decisionpath-table-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar {self.explainer.index_name} para exibir a árvore de decisão",
                                                target="decisionpath-table-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Mostrar árvore:",
                                                id="decisionpath-table-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar a árvore de decisão para exibir o caminho de decisão",
                                                target="decisionpath-table-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="decisionpath-table-highlight-"
                                                + self.name,
                                                options=[
                                                    {"label": str(tree), "value": tree}
                                                    for tree in range(
                                                        self.explainer.no_of_trees
                                                    )
                                                ],
                                                value=self.highlight,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_highlight,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], md=2),
                                    hide=self.hide_selector,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div(id="decisionpath-table-" + self.name),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if args["highlight"] is not None:
            decisionpath_df = self.explainer.get_decisionpath_summary_df(
                int(args["highlight"]), args["index"], pos_label=args["pos_label"]
            )
            html = to_html.table_from_df(decisionpath_df)
        else:
            html = "nenhuma árvore selecionada"
        html = to_html.card(html, title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("decisionpath-table-" + self.name, "children"),
            [
                Input("decisionpath-table-index-" + self.name, "value"),
                Input("decisionpath-table-highlight-" + self.name, "value"),
                Input("pos-label-" + self.name, "value"),
            ],
        )
        def update_decisiontree_table(index, highlight, pos_label):
            if (
                index is None
                or highlight is None
                or not self.explainer.index_exists(index)
            ):
                raise PreventUpdate
            get_decisionpath_df = self.explainer.get_decisionpath_summary_df(
                int(highlight), index, pos_label=pos_label
            )
            return dbc.Table.from_dataframe(get_decisionpath_df)


class DecisionPathGraphComponent(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Gráfico do Caminho de Decisão",
        name=None,
        subtitle="Visualizando a árvore de decisão inteira",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_highlight=False,
        hide_button=False,
        hide_selector=False,
        index_dropdown=True,
        pos_label=None,
        index=None,
        highlight=None,
        description=None,
        **kwargs,
    ):
        """Display dtreeviz decision path

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Decision path graph".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_highlight (bool, optional): hide tree idx selector. Defaults to False.
            hide_button (bool, optional): hide the button, Defaults to False.
            hide_selector (bool, optional): hide pos label selectors. Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            pos_label ({int, str}, optional): initial pos label.
                        Defaults to explainer.pos_label
            index ({str, int}, optional): Initial index to display. Defaults to None.
            highlight ([type], optional): Initial tree idx to display. Defaults to None.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)
        # if explainer.is_regression:
        #     raise ValueError("DecisionPathGraphComponent only available for classifiers for now!")

        self.index_name = "decisionpath-index-" + self.name
        self.highlight_name = "decisionpath-highlight-" + self.name
        if self.description is None:
            self.description = """
        Visualiza o caminho que uma observação percorreu numa árvore de decisão específica,
        mostrando a árvore de decisão inteira e o caminho que essa observação específica
        percorreu.
        """

        self.selector = PosLabelSelector(explainer, name=self.name, pos_label=pos_label)
        self.index_selector = IndexSelector(
            explainer,
            "decisionpath-index-" + self.name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )
        self.register_dependencies("shadow_trees")

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="decisionpath-title-" + self.name
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="decisionpath-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody(
                    [
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                f"{self.explainer.index_name}:",
                                                id="decisionpath-index-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar {self.explainer.index_name} para exibir a árvore de decisão",
                                                target="decisionpath-index-label-"
                                                + self.name,
                                            ),
                                            self.index_selector.layout(),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Mostrar árvore:",
                                                id="decisionpath-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                               f"Selecionar a árvore de decisão para exibir",
                                                target="decisionpath-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="decisionpath-highlight-"
                                                + self.name,
                                                options=[
                                                    {"label": str(tree), "value": tree}
                                                    for tree in range(
                                                        self.explainer.no_of_trees
                                                    )
                                                ],
                                                value=self.highlight,
                                                size="sm",
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_highlight,
                                ),
                                make_hideable(
                                    dbc.Col([self.selector.layout()], width=2),
                                    hide=self.hide_selector,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Gerar Gráfico da Árvore",
                                                color="primary",
                                                id="decisionpath-button-" + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Gerar visualização da árvore de decisão. "
                                                "Só funciona se o graphviz estiver corretamente instalado,"
                                                " e pode demorar algum tempo para árvores grandes.",
                                                target="decisionpath-button-"
                                                + self.name,
                                            ),
                                        ],
                                        md=2,
                                        align="end",
                                    ),
                                    hide=self.hide_button,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Loading(
                                            id="loading-decisionpath-" + self.name,
                                            children=html.Img(
                                                id="decisionpath-svg-" + self.name
                                            ),
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    def component_callbacks(self, app):
        @app.callback(
            Output("decisionpath-svg-" + self.name, "src"),
            [Input("decisionpath-button-" + self.name, "n_clicks")],
            [
                State("decisionpath-index-" + self.name, "value"),
                State("decisionpath-highlight-" + self.name, "value"),
                State("pos-label-" + self.name, "value"),
            ],
        )
        def update_tree_graph(n_clicks, index, highlight, pos_label):
            if index is None or not self.explainer.index_exists(index):
                raise PreventUpdate
            if n_clicks is not None and highlight is not None:
                return self.explainer.decisiontree_encoded(int(highlight), index)
            raise PreventUpdate