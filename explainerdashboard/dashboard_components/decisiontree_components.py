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
        title="Árvores de Decisão", # Translated
        name=None,
        subtitle="Exibindo árvores de decisão individuais", # Translated
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
                        "Árvores de Decisão". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle. Defaults to
                        "Exibindo árvores de decisão individuais". # Updated default
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

        # Translate description based on explainer type
        if isinstance(self.explainer, RandomForestExplainer):
            if self.description is None:
                # Translated description
                self.description = """
            Mostrar a previsão de cada árvore individual numa random forest.
            Isto demonstra como uma random forest é simplesmente uma média de um
            conjunto (ensemble) de árvores de decisão.
            """
            if self.subtitle == "Exibindo árvores de decisão individuais": # Check against translated default
                 # Translated append
                self.subtitle += " dentro da Random Forest"
        elif isinstance(self.explainer, XGBExplainer):
            if self.description is None:
                 # Translated description
                self.description = """
            Mostra as contribuições marginais de cada árvore de decisão num
            conjunto (ensemble) xgboost para a previsão final. Isto demonstra que
            um modelo xgboost é simplesmente uma soma de árvores de decisão individuais.
            """
            if self.subtitle == "Exibindo árvores de decisão individuais": # Check against translated default
                 # Translated append
                self.subtitle += " dentro do modelo xgboost"
        else:
            if self.description is None:
                self.description = "" # Default empty description if type is not RF or XGB

        self.popout = GraphPopout(
            "decisiontrees-" + self.name + "popout",
            "decisiontrees-graph-" + self.name,
            self.title, # uses translated title
            self.description, # uses translated description
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
                                        self.title, # uses translated title
                                        id="decisiontrees-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # uses translated subtitle
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # uses translated description
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
                                                # Translated
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
                                                "Destacar árvore:", # Translated
                                                id="decisiontrees-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                 # Translated
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
            # Assuming explainer.plot_trees generates plots with potentially translated internal labels if needed
            fig = self.explainer.plot_trees(
                args["index"],
                highlight_tree=args["highlight"],
                pos_label=args["pos_label"],
                higher_is_better=self.higher_is_better,
            )
            html = to_html.fig(fig)
        else:
            html = "nenhum índice selecionado" # Translated
        html = to_html.card(html, title=self.title, subtitle=self.subtitle) # uses translated title/subtitle
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
            # Assuming explainer.plot_trees generates plots with potentially translated internal labels if needed
            return self.explainer.plot_trees(
                index,
                highlight_tree=highlight,
                pos_label=pos_label,
                higher_is_better=self.higher_is_better,
            )

        @app.callback(
            Output("decisiontrees-highlight-" + self.name, "value"),
            [Input("decisiontrees-graph-" + self.name, "clickData")],
            # Prevent initial call if not needed, depending on Dash version behavior
            # prevent_initial_call=True
        )
        def update_highlight(clickdata):
            if clickdata is not None and 'points' in clickdata and clickdata['points']:
                 # Assuming the text format remains constant "Tree N: ..." or similar
                 # This parsing logic might need adjustment if the plot text changes
                try:
                    point_text = clickdata["points"][0].get("text", "")
                    # Example parsing logic, adjust based on actual plot text format
                    if "tree no " in point_text:
                         tree_no_str = point_text.split("tree no ")[1].split(":")[0]
                         highlight_tree = int(tree_no_str)
                         return highlight_tree
                    elif "Tree " in point_text: # Alternative common format
                         tree_no_str = point_text.split("Tree ")[1].split(":")[0]
                         highlight_tree = int(tree_no_str)
                         return highlight_tree
                    # Add more parsing logic if needed for different plot formats
                except (IndexError, ValueError, TypeError):
                    # Failed to parse, prevent update
                     pass
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
        title="Tabela do Caminho de Decisão", # Translated
        name=None,
        subtitle="Caminho de decisão através da árvore de decisão", # Translated
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
                        "Tabela do Caminho de Decisão". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle. Defaults to
                        "Caminho de decisão através da árvore de decisão". # Updated default
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
             # Translated description
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
                                        self.title, # uses translated title
                                        id="decisionpath-table-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # uses translated subtitle
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # uses translated description
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
                                                # Translated
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
                                                "Mostrar árvore:", # Translated
                                                id="decisionpath-table-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                # Translated
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
                                        # Container for the table generated by callback
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
        if args["highlight"] is not None and args["index"] is not None:
            # Assuming get_decisionpath_summary_df returns df with English headers
            # Translate headers here if needed for static export
            decisionpath_df = self.explainer.get_decisionpath_summary_df(
                int(args["highlight"]), args["index"], pos_label=args["pos_label"]
            )
            # Example Header translation (adjust based on actual headers):
            # translated_df = decisionpath_df.rename(columns={'feature': 'Variável', 'split': 'Divisão', ...})
            # html = to_html.table_from_df(translated_df)
            html = to_html.table_from_df(decisionpath_df) # Using original for now
        else:
            html = "nenhuma árvore ou índice selecionado" # Translated (modified)
        html = to_html.card(html, title=self.title, subtitle=self.subtitle) # uses translated title/subtitle
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

            try:
                highlight_int = int(highlight) # Ensure highlight is an integer
            except (ValueError, TypeError):
                raise PreventUpdate # Prevent update if highlight is not convertible to int

            # Assuming get_decisionpath_summary_df returns df with English headers
            # Translate headers here if needed for the dynamic table
            decisionpath_df = self.explainer.get_decisionpath_summary_df(
                highlight_int, index, pos_label=pos_label
            )
            # Example Header translation:
            # translated_df = decisionpath_df.rename(columns={'feature': 'Variável', 'split': 'Divisão', ...})
            # return dbc.Table.from_dataframe(translated_df)
            return dbc.Table.from_dataframe(decisionpath_df) # Using original for now


class DecisionPathGraphComponent(ExplainerComponent):
    # Assuming state_props is needed if highlight/index are controlled externally
    _state_props = dict(
        index=("decisionpath-index-", "value"),
        highlight=("decisionpath-highlight-", "value"),
        pos_label=("pos-label-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Gráfico do Caminho de Decisão", # Translated
        name=None,
        subtitle="Visualizando a árvore de decisão inteira", # Translated
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
                        "Gráfico do Caminho de Decisão". # Updated default
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle. Defaults to
                        "Visualizando a árvore de decisão inteira". # Updated default
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

        self.index_name = "decisionpath-index-" + self.name
        self.highlight_name = "decisionpath-highlight-" + self.name
        if self.description is None:
             # Translated description
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
        # Assumes shadow_trees or similar is needed for visualization generation
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
                                        self.title, id="decisionpath-title-" + self.name # uses translated title
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle" # uses translated subtitle
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description, # uses translated description
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
                                                # Translated
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
                                                "Mostrar árvore:", # Translated
                                                id="decisionpath-tree-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                # Translated
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
                                    # Use flex utilities to align button to the right if needed
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                "Gerar Gráfico da Árvore", # Translated
                                                color="primary",
                                                id="decisionpath-button-" + self.name,
                                                n_clicks=0, # Initialize n_clicks
                                                className="mt-2" # Add margin if needed
                                            ),
                                            dbc.Tooltip(
                                                # Translated
                                                "Gerar visualização da árvore de decisão. "
                                                "Só funciona se o graphviz estiver corretamente instalado,"
                                                " e pode demorar algum tempo para árvores grandes.",
                                                target="decisionpath-button-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4, # Adjust width as needed
                                        className="d-flex justify-content-end align-items-center" # Align button
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
                                            type="circle", # Or "default", "cube", "dot"
                                            children=[
                                                # Use html.Iframe for SVG rendering if src provides data URI or URL
                                                # html.Iframe(id="decisionpath-svg-" + self.name, style={'border': 'none', 'width': '100%', 'height': '600px'})
                                                # Or use html.Img if it's just a path to a generated image file (less common for dynamic SVGs)
                                                 html.Img(id="decisionpath-svg-" + self.name, style={'width': '100%'})
                                                ]
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
             # prevent_initial_call=True # Often useful for button clicks
        )
        def update_tree_graph(n_clicks, index, highlight, pos_label):
            # Check if button was clicked
            if n_clicks is None or n_clicks == 0:
                raise PreventUpdate

            # Validate index and highlight
            if index is None or not self.explainer.index_exists(index):
                # Optionally provide feedback to the user here if desired
                # e.g., return a placeholder image or text indicating invalid index
                raise PreventUpdate
            if highlight is None:
                 # Optionally provide feedback
                raise PreventUpdate

            try:
                highlight_int = int(highlight)
            except (ValueError, TypeError):
                # Optionally provide feedback
                raise PreventUpdate

            # Generate the encoded SVG data
            # Make sure explainer.decisiontree_encoded exists and returns a data URI or path
            try:
                svg_data = self.explainer.decisiontree_encoded(highlight_int, index, pos_label=pos_label) # Pass pos_label if needed by method
                if svg_data:
                    return svg_data
                else:
                    # Handle case where generation failed or returned nothing
                    return None # Or return placeholder/error message src
            except Exception as e:
                # Log the error for debugging
                print(f"Error generating decision path graph: {e}")
                # Return placeholder or error indication
                return None # Or a specific error image/message src

            raise PreventUpdate # Fallback
