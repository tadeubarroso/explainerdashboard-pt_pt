__all__ = [
    "RegressionRandomIndexComponent",
    "RegressionPredictionSummaryComponent",
    "PredictedVsActualComponent",
    "ResidualsComponent",
    "RegressionVsColComponent",
    "RegressionModelSummaryComponent",
]

import numpy as np
import pandas as pd

import dash
from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ..dashboard_methods import *
from .. import to_html


class RegressionRandomIndexComponent(ExplainerComponent):
    _state_props = dict(index=("random-index-reg-index-", "value"))

    def __init__(
        self,
        explainer,
        title=None,
        name=None,
        subtitle="Selecione da lista ou escolha aleatoriamente",
        hide_title=False,
        hide_subtitle=False,
        hide_index=False,
        hide_pred_slider=False,
        hide_residual_slider=False,
        hide_pred_or_y=False,
        hide_abs_residuals=False,
        hide_button=False,
        index_dropdown=True,
        index=None,
        pred_slider=None,
        y_slider=None,
        residual_slider=None,
        abs_residual_slider=None,
        pred_or_y="preds",
        abs_residuals=True,
        round=2,
        description=None,
        **kwargs,
    ):
        """Select a random index subject to constraints component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Select Random Index".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_index (bool, optional): Hide index selector.
                        Defaults to False.
            hide_pred_slider (bool, optional): Hide prediction slider.
                        Defaults to False.
            hide_residual_slider (bool, optional): hide residuals slider.
                        Defaults to False.
            hide_pred_or_y (bool, optional): hide prediction or actual toggle.
                        Defaults to False.
            hide_abs_residuals (bool, optional): hide absolute residuals toggle.
                        Defaults to False.
            hide_button (bool, optional): hide button. Defaults to False.
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            index ({str, int}, optional): Initial index to display.
                        Defaults to None.
            pred_slider ([lb, ub], optional): Initial values for prediction
                        values slider [lowerbound, upperbound]. Defaults to None.
            y_slider ([lb, ub], optional): Initial values for y slider
                        [lower bound, upper bound]. Defaults to None.
            residual_slider ([lb, ub], optional): Initial values for residual slider
                        [lower bound, upper bound]. Defaults to None.
            abs_residual_slider ([lb, ub], optional): Initial values for absolute
                        residuals slider [lower bound, upper bound]
                        Defaults to None.
            pred_or_y (str, {'preds', 'y'}, optional): Initial use predictions
                        or y slider. Defaults to "preds".
            abs_residuals (bool, optional): Initial use residuals or absolute
                        residuals. Defaults to True.
            round (int, optional): rounding used for slider spacing. Defaults to 2.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title or f"Selecione o {explainer.index_name}", name)
        assert self.explainer.is_regression, (
            "explainer não é um RegressionExplainer, pelo que o RegressionRandomIndexComponent "
            "não funcionará. Tente usar o ClassifierRandomIndexComponent em vez disso."
        )

        # if self.title is None:
        #     self.title = f"Select {self.explainer.index_name}"

        self.index_name = "random-index-reg-index-" + self.name
        self.index_selector = IndexSelector(
            explainer,
            self.index_name,
            index=index,
            index_dropdown=index_dropdown,
            **kwargs,
        )

        if self.explainer.y_missing:
            self.hide_residual_slider = True
            self.hide_pred_or_y = True
            self.hide_abs_residuals = True
            self.pred_or_y = "preds"
            self.y_slider = [0.0, 1.0]
            self.residual_slider = [0.0, 1.0]
            self.abs_residual_slider = [0.0, 1.0]

        if self.pred_slider is None:
            self.pred_slider = [
                float(self.explainer.preds.min()),
                float(self.explainer.preds.max()),
            ]

        if not self.explainer.y_missing:
            if self.y_slider is None:
                self.y_slider = [
                    float(self.explainer.y.min()),
                    float(self.explainer.y.max()),
                ]

            if self.residual_slider is None:
                self.residual_slider = [
                    float(self.explainer.residuals.min()),
                    float(self.explainer.residuals.max()),
                ]

            if self.abs_residual_slider is None:
                self.abs_residual_slider = [
                    float(self.explainer.abs_residuals.min()),
                    float(self.explainer.abs_residuals.max()),
                ]

            assert (
                len(self.pred_slider) == 2
                and self.pred_slider[0] <= self.pred_slider[1]
            ), "pred_slider should be a list of a [lower_bound, upper_bound]!"

            assert (
                len(self.y_slider) == 2 and self.y_slider[0] <= self.y_slider[1]
            ), "y_slider should be a list of a [lower_bound, upper_bound]!"

            assert (
                len(self.residual_slider) == 2
                and self.residual_slider[0] <= self.residual_slider[1]
            ), "residual_slider should be a list of a [lower_bound, upper_bound]!"

            assert (
                len(self.abs_residual_slider) == 2
                and self.abs_residual_slider[0] <= self.abs_residual_slider[1]
            ), "abs_residual_slider should be a list of a [lower_bound, upper_bound]!"

        self.y_slider = [float(y) for y in self.y_slider]
        self.pred_slider = [float(p) for p in self.pred_slider]
        self.residual_slider = [float(r) for r in self.residual_slider]
        self.abs_residual_slider = [float(a) for a in self.abs_residual_slider]

        assert self.pred_or_y in {
            "preds",
            "y",
        }, "pred_or_y should be in ['preds', 'y']!"

        if self.description is None:
            self.description = f"""
        Pode selecionar um {self.explainer.index_name} diretamente escolhendo-o
        na lista pendente (se começar a escrever, pode pesquisar dentro da lista),
        ou clique no botão '{self.explainer.index_name} Aleatório' para selecionar aleatoriamente
        um {self.explainer.index_name} que cumpra as restrições. Por exemplo,
        pode selecionar um {self.explainer.index_name} com um {self.explainer.target} previsto
        muito alto, ou um {self.explainer.target} observado muito baixo,
        ou um {self.explainer.index_name} cujo {self.explainer.target} previsto
        estava muito longe do {self.explainer.target} observado e, por isso, tinha um
        resíduo (absoluto) alto.
        """

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
                                        id="random-index-reg-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="random-index-reg-title-" + self.name,
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
                                    dbc.Col([self.index_selector.layout()], md=8),
                                    hide=self.hide_index,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Button(
                                                f"Aleatório {self.explainer.index_name}",
                                                color="primary",
                                                id="random-index-reg-button-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecionar um {self.explainer.index_name} aleatório de acordo com as restrições",
                                                target="random-index-reg-button-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_button,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo previsto:",
                                                        id="random-index-reg-pred-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-pred-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"{self.explainer.target} previsto estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-pred-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-pred-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.preds.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.preds.max()
                                                        ),
                                                        step=np.float_power(
                                                            10, -self.round
                                                        ),
                                                        value=[
                                                            self.pred_slider[0],
                                                            self.pred_slider[1],
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.preds.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.preds.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.preds.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.preds.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-pred-slider-div-"
                                                + self.name,
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo observado:",
                                                        id="random-index-reg-y-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-y-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"{self.explainer.target} observado estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-y-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-y-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.y.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.y.max()
                                                        ),
                                                        step=np.float_power(
                                                            10, -self.round
                                                        ),
                                                        value=[
                                                            self.y_slider[0],
                                                            self.y_slider[1],
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.y.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.y.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.y.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.y.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-y-slider-div-"
                                                + self.name,
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    hide=self.hide_pred_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Intervalo:",
                                                id="random-index-reg-preds-or-y-label-"
                                                + self.name,
                                                html_for="random-index-reg-preds-or-y-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-reg-preds-or-y-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Predicted",
                                                        "value": "preds",
                                                    },
                                                    {"label": "Observed", "value": "y"},
                                                ],
                                                value=self.pred_or_y,
                                            ),
                                            dbc.Tooltip(
                                                f"Pode selecionar um  {self.explainer.index_name} aleatório "
                                                f"apenas dentro de um certo intervalo do {self.explainer.target} observado ou "
                                                f"dentro de um certo intervalo do {self.explainer.target} previsto.",
                                                target="random-index-reg-preds-or-y-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_pred_or_y,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Intervalo de resíduos:",
                                                        id="random-index-reg-residual-slider-label-"
                                                        + self.name,
                                                        html_for="random-index-reg-residual-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o "
                                                        f"resíduo (diferença entre o {self.explainer.target} observado e o {self.explainer.target} previsto)"
                                                        " estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-residual-slider-label-"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-residual-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.residuals.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.residuals.max()
                                                        ),
                                                        step=float(
                                                            np.float_power(
                                                                10, -self.round
                                                            )
                                                        ),
                                                        value=[
                                                            float(
                                                                self.residual_slider[0]
                                                            ),
                                                            float(
                                                                self.residual_slider[1]
                                                            ),
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.residuals.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.residuals.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.residuals.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.residuals.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-residual-slider-div-"
                                                + self.name,
                                            ),
                                            html.Div(
                                                [
                                                    dbc.Label(
                                                        "Resíduos absolutos",
                                                        id="random-index-reg-abs-residual-slider-label"
                                                        + self.name,
                                                        html_for="random-index-reg-abs-residual-slider-"
                                                        + self.name,
                                                    ),
                                                    dbc.Tooltip(
                                                        f"Selecionar apenas {self.explainer.index_name} onde o resíduo absoluto "
                                                        f"(diferença entre o {self.explainer.target} observado e o {self.explainer.target} previsto)"
                                                        " estava dentro do seguinte intervalo:",
                                                        target="random-index-reg-abs-residual-slider-label"
                                                        + self.name,
                                                    ),
                                                    dcc.RangeSlider(
                                                        id="random-index-reg-abs-residual-slider-"
                                                        + self.name,
                                                        min=float(
                                                            self.explainer.abs_residuals.min()
                                                        ),
                                                        max=float(
                                                            self.explainer.abs_residuals.max()
                                                        ),
                                                        step=float(
                                                            np.float_power(
                                                                10, -self.round
                                                            )
                                                        ),
                                                        value=[
                                                            float(
                                                                self.abs_residual_slider[
                                                                    0
                                                                ]
                                                            ),
                                                            float(
                                                                self.abs_residual_slider[
                                                                    1
                                                                ]
                                                            ),
                                                        ],
                                                        marks={
                                                            float(
                                                                self.explainer.abs_residuals.min()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.abs_residuals.min(),
                                                                    self.round,
                                                                )
                                                            ),
                                                            float(
                                                                self.explainer.abs_residuals.max()
                                                            ): str(
                                                                np.round(
                                                                    self.explainer.abs_residuals.max(),
                                                                    self.round,
                                                                )
                                                            ),
                                                        },
                                                        allowCross=False,
                                                        tooltip={
                                                            "always_visible": False
                                                        },
                                                    ),
                                                ],
                                                id="random-index-reg-abs-residual-slider-div-"
                                                + self.name,
                                            ),
                                        ],
                                        md=8,
                                    ),
                                    hide=self.hide_residual_slider,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Resíduos:",
                                                id="random-index-reg-abs-residual-label-"
                                                + self.name,
                                                html_for="random-index-reg-abs-residual-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="random-index-reg-abs-residual-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Resíduos",
                                                        "value": "relative",
                                                    },
                                                    {
                                                        "label": "Resíduos Absolutos",
                                                        "value": "absolute",
                                                    },
                                                ],
                                                value="absolute"
                                                if self.abs_residuals
                                                else "relative",
                                            ),
                                            dbc.Tooltip(
                                                f"Pode selecionar um {self.explainer.index_name} aleatório apenas "
                                                f"dentro de um certo intervalo de resíduos "
                                                f"(diferença entre o {self.explainer.target} observado e previsto), "
                                                f"por exemplo, apenas {self.explainer.index_name} para os quais a previsão "
                                                f"foi demasiado alta ou demasiado baixa."
                                                f"Ou pode selecionar apenas dentro de um certo intervalo de resíduos absolutos. Por "
                                                f"exemplo, selecionar apenas {self.explainer.index_name} para os quais a previsão estava errada "
                                                f"por pelo menos uma certa quantidade de {self.explainer.units}.",
                                                target="random-index-reg-abs-residual-label-"
                                                + self.name,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_abs_residuals,
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)

        html = to_html.card(
            f"Índice selecionado: <b>{self.explainer.get_index(args['index'])}</b>",
            title=self.title,
        )
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("random-index-reg-pred-slider-div-" + self.name, "style"),
                Output("random-index-reg-y-slider-div-" + self.name, "style"),
            ],
            [Input("random-index-reg-preds-or-y-" + self.name, "value")],
        )
        def update_reg_hidden_div_pred_sliders(preds_or_y):
            if preds_or_y == "preds":
                return (None, dict(display="none"))
            elif preds_or_y == "y":
                return (dict(display="none"), None)
            raise PreventUpdate

        @app.callback(
            [
                Output("random-index-reg-residual-slider-div-" + self.name, "style"),
                Output(
                    "random-index-reg-abs-residual-slider-div-" + self.name, "style"
                ),
            ],
            [Input("random-index-reg-abs-residual-" + self.name, "value")],
        )
        def update_reg_hidden_div_pred_sliders(abs_residuals):
            if abs_residuals == "absolute":
                return (dict(display="none"), None)
            else:
                return (None, dict(display="none"))
            raise PreventUpdate

        @app.callback(
            [
                Output("random-index-reg-residual-slider-" + self.name, "min"),
                Output("random-index-reg-residual-slider-" + self.name, "max"),
                Output("random-index-reg-residual-slider-" + self.name, "value"),
                Output("random-index-reg-residual-slider-" + self.name, "marks"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "min"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "max"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "value"),
                Output("random-index-reg-abs-residual-slider-" + self.name, "marks"),
            ],
            [
                Input("random-index-reg-pred-slider-" + self.name, "value"),
                Input("random-index-reg-y-slider-" + self.name, "value"),
            ],
            [
                State("random-index-reg-preds-or-y-" + self.name, "value"),
                State("random-index-reg-residual-slider-" + self.name, "value"),
                State("random-index-reg-abs-residual-slider-" + self.name, "value"),
            ],
        )
        def update_residual_slider_limits(
            pred_range, y_range, preds_or_y, residuals_range, abs_residuals_range
        ):
            if preds_or_y == "preds":
                min_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].min()
                )
                max_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].max()
                )
                min_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].min()
                )
                max_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.preds >= pred_range[0])
                        & (self.explainer.preds <= pred_range[1])
                    ].max()
                )
            elif preds_or_y == "y":
                min_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].min()
                )
                max_residuals = float(
                    self.explainer.residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].max()
                )
                min_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].min()
                )
                max_abs_residuals = float(
                    self.explainer.abs_residuals[
                        (self.explainer.y >= y_range[0])
                        & (self.explainer.y <= y_range[1])
                    ].max()
                )

            new_residuals_range = [
                max(min_residuals, residuals_range[0]),
                min(max_residuals, residuals_range[1]),
            ]
            new_abs_residuals_range = [
                max(min_abs_residuals, abs_residuals_range[0]),
                min(max_abs_residuals, abs_residuals_range[1]),
            ]
            residuals_marks = {
                min_residuals: str(np.round(min_residuals, self.round)),
                max_residuals: str(np.round(max_residuals, self.round)),
            }
            abs_residuals_marks = {
                min_abs_residuals: str(np.round(min_abs_residuals, self.round)),
                max_abs_residuals: str(np.round(max_abs_residuals, self.round)),
            }
            return (
                min_residuals,
                max_residuals,
                new_residuals_range,
                residuals_marks,
                min_abs_residuals,
                max_abs_residuals,
                new_abs_residuals_range,
                abs_residuals_marks,
            )

        @app.callback(
            Output("random-index-reg-index-" + self.name, "value"),
            [Input("random-index-reg-button-" + self.name, "n_clicks")],
            [
                State("random-index-reg-pred-slider-" + self.name, "value"),
                State("random-index-reg-y-slider-" + self.name, "value"),
                State("random-index-reg-residual-slider-" + self.name, "value"),
                State("random-index-reg-abs-residual-slider-" + self.name, "value"),
                State("random-index-reg-preds-or-y-" + self.name, "value"),
                State("random-index-reg-abs-residual-" + self.name, "value"),
            ],
        )
        def update_index(
            n_clicks,
            pred_range,
            y_range,
            residual_range,
            abs_residuals_range,
            preds_or_y,
            abs_residuals,
        ):
            triggers = [
                trigger["prop_id"] for trigger in dash.callback_context.triggered
            ]
            if f"random-index-reg-button-{self.name}.n_clicks" not in triggers:
                raise PreventUpdate
            if n_clicks is None and self.index is not None:
                raise PreventUpdate
            if preds_or_y == "preds":
                if abs_residuals == "absolute":
                    return self.explainer.random_index(
                        pred_min=pred_range[0],
                        pred_max=pred_range[1],
                        abs_residuals_min=abs_residuals_range[0],
                        abs_residuals_max=abs_residuals_range[1],
                        return_str=True,
                    )
                else:
                    return self.explainer.random_index(
                        pred_min=pred_range[0],
                        pred_max=pred_range[1],
                        residuals_min=residual_range[0],
                        residuals_max=residual_range[1],
                        return_str=True,
                    )
            elif preds_or_y == "y":
                if abs_residuals == "absolute":
                    return self.explainer.random_index(
                        y_min=y_range[0],
                        y_max=y_range[1],
                        abs_residuals_min=abs_residuals_range[0],
                        abs_residuals_max=abs_residuals_range[1],
                        return_str=True,
                    )
                else:
                    return self.explainer.random_index(
                        y_min=pred_range[0],
                        y_max=pred_range[1],
                        residuals_min=residual_range[0],
                        residuals_max=residual_range[1],
                        return_str=True,
                    )


class RegressionModelSummaryComponent(ExplainerComponent):
    def __init__(
        self,
        explainer,
        title="Sumário do Modelo",
        name=None,
        subtitle="Métricas quantitativas para o desempenho do modelo",
        hide_title=False,
        hide_subtitle=False,
        round=3,
        show_metrics=None,
        description=None,
        **kwargs,
    ):
        """Show model summary statistics (RMSE, MAE, R2) component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Model Summary".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional): hide title
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            round (int): rounding to perform to metric floats.
            show_metrics (List): list of metrics to display in order. Defaults
                to None, displaying all metrics.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)
        if self.description is None:
            self.description = f"""
        Na tabela abaixo pode encontrar várias métricas de desempenho de regressão
        que descrevem quão bem o modelo consegue prever
        {self.explainer.target}.
        """
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        metrics_dict = self.explainer.metrics_descriptions()
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(show_metrics=self.show_metrics), index=["Score"]
            )
            .T.rename_axis(index="metric")
            .reset_index()
            .round(self.round)
        )
        metrics_table = dbc.Table.from_dataframe(
            metrics_df, striped=False, bordered=False, hover=False
        )
        metrics_table, tooltips = get_dbc_tooltips(
            metrics_table, metrics_dict, "reg-model-summary-div-hover", self.name
        )
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title,
                                        id="reg-model-summary-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="reg-model-summary-title-" + self.name,
                                    ),
                                ]
                            ),
                        ]
                    ),
                    hide=self.hide_title,
                ),
                dbc.CardBody([metrics_table, *tooltips]),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        metrics_df = (
            pd.DataFrame(
                self.explainer.metrics(show_metrics=self.show_metrics), index=["Score"]
            )
            .T.rename_axis(index="metric")
            .reset_index()
            .round(self.round)
        )
        html = to_html.table_from_df(metrics_df)
        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html


class RegressionPredictionSummaryComponent(ExplainerComponent):
    _state_props = dict(index=("reg-prediction-index-", "value"))

    def __init__(
        self,
        explainer,
        title="Prediction",
        name=None,
        hide_index=False,
        hide_title=False,
        hide_subtitle=False,
        hide_table=False,
        index_dropdown=True,
        feature_input_component=None,
        index=None,
        round=3,
        description=None,
        **kwargs,
    ):
        """Shows a summary for a particular prediction

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Prediction".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            hide_index (bool, optional): hide index selector. Defaults to False.
            hide_title (bool, optional): hide title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_table (bool, optional): hide the results table
            index_dropdown (bool, optional): Use dropdown for index input instead
                of free text input. Defaults to True.
            feature_input_component (FeatureInputComponent): A FeatureInputComponent
                that will give the input to the graph instead of the index selector.
                If not None, hide_index=True. Defaults to None.
            index ({int, str}, optional): Index to display prediction summary for. Defaults to None.
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.index_name = "reg-prediction-index-" + self.name
        self.index_selector = IndexSelector(
            explainer, self.index_name, index=index, index_dropdown=index_dropdown
        )

        if self.feature_input_component is not None:
            self.exclude_callbacks(self.feature_input_component)
            self.hide_index = True

        if self.description is None:
            self.description = f"""
        Shows the predicted {self.explainer.target} and the observed {self.explainer.target},
        as well as the difference between the two (residual)
        """

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.H3(
                                self.title,
                                id="reg-prediction-title-" + self.name,
                                className="card-title",
                            ),
                            dbc.Tooltip(
                                self.description,
                                target="reg-prediction-title-" + self.name,
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
                                            dbc.Label(f"{self.explainer.index_name}:"),
                                            self.index_selector.layout(),
                                        ],
                                        md=6,
                                    ),
                                    hide=self.hide_index,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [dbc.Col([html.Div(id="reg-prediction-div-" + self.name)])]
                        ),
                    ]
                ),
            ]
        )

    def get_state_tuples(self):
        _state_tuples = super().get_state_tuples()
        if self.feature_input_component is not None:
            _state_tuples.extend(self.feature_input_component.get_state_tuples())
        return sorted(list(set(_state_tuples)))

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if self.feature_input_component is None:
            if args["index"] is not None:
                preds_df = self.explainer.prediction_result_df(
                    args["index"], round=self.round
                )
                html = to_html.table_from_df(preds_df)
            else:
                html = "no index selected"
        else:
            inputs = {
                k: v
                for k, v in self.feature_input_component.get_state_args(
                    state_dict
                ).items()
                if k != "index"
            }
            inputs = list(inputs.values())
            if len(inputs) == len(
                self.feature_input_component._input_features
            ) and not any([i is None for i in inputs]):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round
                )
                html = to_html.table_from_df(preds_df)
            else:
                html = f"<div>dados de entrada incorretos</div>"

        html = to_html.card(html, title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        if self.feature_input_component is None:

            @app.callback(
                Output("reg-prediction-div-" + self.name, "children"),
                [Input("reg-prediction-index-" + self.name, "value")],
            )
            def update_output_div(index):
                if index is None or not self.explainer.index_exists(index):
                    raise PreventUpdate
                preds_df = self.explainer.prediction_result_df(index, round=self.round)
                return make_hideable(
                    dbc.Table.from_dataframe(
                        preds_df, striped=False, bordered=False, hover=False
                    ),
                    hide=self.hide_table,
                )

        else:

            @app.callback(
                Output("reg-prediction-div-" + self.name, "children"),
                [*self.feature_input_component._feature_callback_inputs],
            )
            def update_output_div(*inputs):
                X_row = self.explainer.get_row_from_input(inputs, ranked_by_shap=True)
                preds_df = self.explainer.prediction_result_df(
                    X_row=X_row, round=self.round
                )
                return make_hideable(
                    dbc.Table.from_dataframe(
                        preds_df, striped=False, bordered=False, hover=False
                    ),
                    hide=self.hide_table,
                )


class PredictedVsActualComponent(ExplainerComponent):
    _state_props = dict(
        log_x=("pred-vs-actual-logx-", "value"), log_y=("pred-vs-actual-logy-", "value")
    )

    def __init__(
        self,
        explainer,
        title="Previsto vs. Observado",
        name=None,
        subtitle="Quão próximo está o valor previsto do observado?",
        hide_title=False,
        hide_subtitle=False,
        hide_log_x=False,
        hide_log_y=False,
        hide_popout=False,
        logs=False,
        log_x=False,
        log_y=False,
        round=3,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Shows a plot of predictions vs y.

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Predicted vs Actual".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_log_x (bool, optional): Hide the log_x toggle. Defaults to False.
            hide_log_y (bool, optional): Hide the log_y toggle. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            logs (bool, optional): Whether to use log axis. Defaults to False.
            log_x (bool, optional): log only x axis. Defaults to False.
            log_y (bool, optional): log only y axis. Defaults to False.
            round (int, optional): rounding to apply to float predictions.
                Defaults to 3.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        self.logs, self.log_x, self.log_y = logs, log_x, log_y

        if self.description is None:
            self.description = f"""
        O gráfico mostra o {self.explainer.target} observado e o {self.explainer.target} previsto
        no mesmo gráfico. Um modelo perfeito teria
        todos os pontos na diagonal (previsto igual a observado). Quanto mais
        afastados os pontos estiverem da diagonal, pior é o modelo a prever
        {self.explainer.target}.
        """

        self.popout = GraphPopout(
            "pred-vs-actual-" + self.name + "popout",
            "pred-vs-actual-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds"])

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
                                        id="pred-vs-actual-title-" + self.name,
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="pred-vs-actual-title-" + self.name,
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
                                            dbc.Row(
                                                [
                                                    # html.Label("Log y"),
                                                    dbc.RadioButton(
                                                        id="pred-vs-actual-logy-"
                                                        + self.name,
                                                        className="form-check-input",
                                                        value=self.log_y,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Ao usar um eixo logarítmico, é mais fácil ver erros relativos "
                                                        "em vez de erros absolutos.",
                                                        target="pred-vs-actual-logy-"
                                                        + self.name,
                                                    ),
                                                    dbc.Label(
                                                        "Log y",
                                                        html_for="pred-vs-actual-logy-"
                                                        + self.name,
                                                        className="form-check-label",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=1,
                                        align="center",
                                    ),
                                    hide=self.hide_log_y,
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="pred-vs-actual-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
                                        ),
                                    ],
                                    md=11,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.RadioButton(
                                                        id="pred-vs-actual-logx-"
                                                        + self.name,
                                                        className="form-check-input",
                                                        value=self.log_x,
                                                    ),
                                                    dbc.Tooltip(
                                                        "Ao usar um eixo logarítmico, é mais fácil ver erros relativos "
                                                        "em vez de erros absolutos.",
                                                        target="pred-vs-actual-logx-"
                                                        + self.name,
                                                    ),
                                                    dbc.Label(
                                                        "Log x",
                                                        html_for="pred-vs-actual-logx-"
                                                        + self.name,
                                                        className="form-check-label",
                                                        size="sm",
                                                    ),
                                                ]
                                            ),
                                        ],
                                        md=2,
                                    ),
                                    hide=self.hide_log_x,
                                ),
                            ],
                            justify="center",
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
        fig = self.explainer.plot_predicted_vs_actual(
            log_x=bool(args["log_x"]),
            log_y=bool(args["log_y"]),
            round=self.round,
            plot_sample=self.plot_sample,
        )
        html = to_html.card(to_html.fig(fig), title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("pred-vs-actual-graph-" + self.name, "figure"),
            [
                Input("pred-vs-actual-logx-" + self.name, "value"),
                Input("pred-vs-actual-logy-" + self.name, "value"),
            ],
        )
        def update_predicted_vs_actual_graph(log_x, log_y):
            return self.explainer.plot_predicted_vs_actual(
                log_x=log_x, log_y=log_y, round=self.round, plot_sample=self.plot_sample
            )


class ResidualsComponent(ExplainerComponent):
    _state_props = dict(
        pred_or_actual=("residuals-pred-or-actual-", "value"),
        residuals=("residuals-type-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Residuals",
        name=None,
        subtitle="Qual o desvio do modelo?",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_pred_or_actual=False,
        hide_ratio=False,
        hide_popout=False,
        pred_or_actual="vs_pred",
        residuals="difference",
        round=3,
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Residuals plot component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Residuals".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_pred_or_actual (bool, optional): hide vs predictions or vs
                        actual for x-axis toggle. Defaults to False.
            hide_ratio (bool, optional): hide residual type dropdown. Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            pred_or_actual (str, {'vs_actual', 'vs_pred'}, optional): Whether
                        to plot actual or predictions on the x-axis.
                        Defaults to "vs_pred".
            residuals (str, {'difference', 'ratio', 'log-ratio'} optional):
                    How to calcualte residuals. Defaults to 'difference'.
            round (int, optional): rounding to apply to float predictions.
                Defaults to 3.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        assert residuals in ["difference", "ratio", "log-ratio"], (
            "o parâmetro residuals deve estar em ['difference', 'ratio', 'log-ratio']"
            f" mas passou residuals={residuals}"
        )

        if self.description is None:
            self.description = f"""
        Os resíduos são a diferença entre o {self.explainer.target} observado
        e o {self.explainer.target} previsto. Neste gráfico, pode verificar se
        os resíduos são maiores ou menores para resultados observados/previstos mais altos/baixos.
        Assim, pode verificar se o modelo funciona melhor ou worse para diferentes níveis de {self.explainer.target}.
        """

        self.popout = GraphPopout(
            "residuals-" + self.name + "popout",
            "residuals-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="residuals-title-" + self.name
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="residuals-title-" + self.name,
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
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="residuals-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
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
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Eixo horizontal:",
                                                            html_for="residuals-pred-or-actual-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            options=[
                                                                {
                                                                    "label": "Previsto",
                                                                    "value": "vs_pred",
                                                                },
                                                                {
                                                                    "label": "Observado",
                                                                    "value": "vs_actual",
                                                                },
                                                            ],
                                                            value=self.pred_or_actual,
                                                            id="residuals-pred-or-actual-"
                                                            + self.name,
                                                            size="sm",
                                                        ),
                                                    ],
                                                    id="residuals-pred-or-actual-form-"
                                                    + self.name,
                                                ),
                                                dbc.Tooltip(
                                                    "Selecione o que gostaria de colocar no eixo x:"
                                                    f" {self.explainer.target} observado ou {self.explainer.target} previsto.",
                                                    target="residuals-pred-or-actual-form-"
                                                    + self.name,
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        hide=self.hide_pred_or_actual,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Tipo de resíduo:",
                                                            id="residuals-type-label-"
                                                            + self.name,
                                                            html_for="residuals-type-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Tipo de resíduos a exibir: y-preds (diferença), "
                                                            "y/previsto (rácio) or log(y/preds) (log-rácio).",
                                                            target="residuals-type-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="residuals-type-"
                                                            + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Diferença",
                                                                    "value": "difference",
                                                                },
                                                                {
                                                                    "label": "Rácio",
                                                                    "value": "ratio",
                                                                },
                                                                {
                                                                    "label": "Log-Rácio",
                                                                    "value": "log-ratio",
                                                                },
                                                            ],
                                                            value=self.residuals,
                                                            size="sm",
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            md=3,
                                        ),
                                        hide=self.hide_ratio,
                                    ),
                                ],
                                justify="evenly",
                            ),
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        vs_actual = args["pred_or_actual"] == "vs_actual"
        fig = self.explainer.plot_residuals(
            vs_actual=vs_actual,
            residuals=args["residuals"],
            round=self.round,
            plot_sample=self.plot_sample,
        )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        html = to_html.card(to_html.fig(fig), title=self.title)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            Output("residuals-graph-" + self.name, "figure"),
            [
                Input("residuals-pred-or-actual-" + self.name, "value"),
                Input("residuals-type-" + self.name, "value"),
            ],
        )
        def update_residuals_graph(pred_or_actual, residuals):
            vs_actual = pred_or_actual == "vs_actual"
            return self.explainer.plot_residuals(
                vs_actual=vs_actual,
                residuals=residuals,
                round=self.round,
                plot_sample=self.plot_sample,
            )


class RegressionVsColComponent(ExplainerComponent):
    _state_props = dict(
        col=("reg-vs-col-col-", "value"),
        display=("reg-vs-col-display-type-", "value"),
        points=("reg-vs-col-show-points-", "value"),
        winsor=("reg-vs-col-winsor-", "value"),
        cats_topx=("reg-vs-col-n-categories-", "value"),
        cats_sort=("reg-vs-col-categories-sort-", "value"),
    )

    def __init__(
        self,
        explainer,
        title="Gráfico vs característica",
        name=None,
        subtitle="As previsões e os resíduos estão correlacionados com as características?",
        hide_title=False,
        hide_subtitle=False,
        hide_footer=False,
        hide_col=False,
        hide_ratio=False,
        hide_points=False,
        hide_winsor=False,
        hide_cats_topx=False,
        hide_cats_sort=False,
        hide_popout=False,
        col=None,
        display="difference",
        round=3,
        points=True,
        winsor=0,
        cats_topx=10,
        cats_sort="freq",
        plot_sample=None,
        description=None,
        **kwargs,
    ):
        """Show residuals, observed or preds vs a particular Feature component

        Args:
            explainer (Explainer): explainer object constructed with either
                        ClassifierExplainer() or RegressionExplainer()
            title (str, optional): Title of tab or page. Defaults to
                        "Plot vs feature".
            name (str, optional): unique name to add to Component elements.
                        If None then random uuid is generated to make sure
                        it's unique. Defaults to None.
            subtitle (str): subtitle
            hide_title (bool, optional) Hide the title. Defaults to False.
            hide_subtitle (bool, optional): Hide subtitle. Defaults to False.
            hide_footer (bool, optional): hide the footer at the bottom of the component
            hide_col (bool, optional): Hide de column selector. Defaults to False.
            hide_ratio (bool, optional): Hide the  toggle. Defaults to False.
            hide_points (bool, optional): Hide group points toggle. Defaults to False.
            hide_winsor (bool, optional): Hide winsor input. Defaults to False.
            hide_cats_topx (bool, optional): hide the categories topx input. Defaults to False.
            hide_cats_sort (bool, optional): hide the categories sort selector.Defaults to False.
            hide_popout (bool, optional): hide popout button. Defaults to False.
            col ([type], optional): Initial feature to display. Defaults to None.
            display (str, {'observed', 'predicted', difference', 'ratio', 'log-ratio'} optional):
                    What to display on y axis. Defaults to 'difference'.
            round (int, optional): rounding to apply to float predictions.
                Defaults to 3.
            points (bool, optional): display point cloud next to violin plot
                    for categorical cols. Defaults to True
            winsor (int, 0-50, optional): percentage of outliers to winsor out of
                    the y-axis. Defaults to 0.
            cats_topx (int, optional): maximum number of categories to display
                for categorical features. Defaults to 10.
            cats_sort (str, optional): how to sort categories: 'alphabet',
                'freq' or 'shap'. Defaults to 'freq'.
            plot_sample (int, optional): Instead of all points only plot a random
                sample of points. Defaults to None (=all points)
            description (str, optional): Tooltip to display when hover over
                component title. When None default text is shown.
        """
        super().__init__(explainer, title, name)

        if self.col is None:
            self.col = self.explainer.columns_ranked_by_shap()[0]

        assert self.display in {
            "observed",
            "predicted",
            "difference",
            "ratio",
            "log-ratio",
        }, (
            "parameter display should in {'observed', 'predicted', 'difference', 'ratio', 'log-ratio'}"
            f" but you passed display={self.display}!"
        )

        if self.description is None:
            self.description = f"""
        Este gráfico mostra os resíduos (diferença entre o {self.explainer.target} observado
        e o {self.explainer.target} previsto) plotados contra os valores de diferentes características,
        ou o {self.explainer.target} observado ou previsto.
        Isto permite inspecionar se o modelo está mais errado para intervalos específicos
        de valores de características do que para outros.
        """
        self.popout = GraphPopout(
            "reg-vs-col-" + self.name + "popout",
            "reg-vs-col-graph-" + self.name,
            self.title,
            self.description,
        )
        self.register_dependencies(["preds", "residuals"])

    def layout(self):
        return dbc.Card(
            [
                make_hideable(
                    dbc.CardHeader(
                        [
                            html.Div(
                                [
                                    html.H3(
                                        self.title, id="reg-vs-col-title-" + self.name
                                    ),
                                    make_hideable(
                                        html.H6(
                                            self.subtitle, className="card-subtitle"
                                        ),
                                        hide=self.hide_subtitle,
                                    ),
                                    dbc.Tooltip(
                                        self.description,
                                        target="reg-vs-col-title-" + self.name,
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
                                                "Característica:",
                                                id="reg-vs-col-col-label-" + self.name,
                                            ),
                                            dbc.Tooltip(
                                                "Selecione a característica a exibir no eixo x.",
                                                target="reg-vs-col-col-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="reg-vs-col-col-" + self.name,
                                                options=[
                                                    {"label": col, "value": col}
                                                    for col in self.explainer.columns_ranked_by_shap()
                                                ],
                                                value=self.col,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_col,
                                ),
                                make_hideable(
                                    dbc.Col(
                                        [
                                            dbc.Label(
                                                "Exibir:",
                                                id="reg-vs-col-display-type-label-"
                                                + self.name,
                                            ),
                                            dbc.Tooltip(
                                                f"Selecione o que exibir no eixo y: {self.explainer.target} observado, "
                                                f"{self.explainer.target} previsto ou resíduos. Os resíduos podem ser "
                                                "calculados pela diferença (y-previsto), "
                                                "rácio (y/previsto) ou log-rácio log(y/previsto). Este último facilita a "
                                                "visualização de diferenças relativas.",
                                                target="reg-vs-col-display-type-label-"
                                                + self.name,
                                            ),
                                            dbc.Select(
                                                id="reg-vs-col-display-type-"
                                                + self.name,
                                                options=[
                                                    {
                                                        "label": "Observado",
                                                        "value": "observed",
                                                    },
                                                    {
                                                        "label": "Previsto",
                                                        "value": "predicted",
                                                    },
                                                    {
                                                        "label": "Resíduos: Diferença",
                                                        "value": "difference",
                                                    },
                                                    {
                                                        "label": "Resíduos: Rácio",
                                                        "value": "ratio",
                                                    },
                                                    {
                                                        "label": "Resíduos: Log-Rácio",
                                                        "value": "log-ratio",
                                                    },
                                                ],
                                                value=self.display,
                                            ),
                                        ],
                                        md=4,
                                    ),
                                    hide=self.hide_ratio,
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dcc.Graph(
                                            id="reg-vs-col-graph-" + self.name,
                                            config=dict(
                                                modeBarButtons=[["toImage"]],
                                                displaylogo=False,
                                            ),
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
                make_hideable(
                    dbc.CardFooter(
                        [
                            dbc.Row(
                                [
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                dbc.Label(
                                                    "Winsor:",
                                                    id="reg-vs-col-winsor-label-"
                                                    + self.name,
                                                ),
                                                dbc.Tooltip(
                                                    "Exclui os valores y mais altos e mais baixos do gráfico. "
                                                    "Quando existem alguns outliers reais, pode ajudar removê-los"
                                                    " do gráfico para que seja mais fácil ver o padrão geral.",
                                                    target="reg-vs-col-winsor-label-"
                                                    + self.name,
                                                ),
                                                dbc.Input(
                                                    id="reg-vs-col-winsor-" + self.name,
                                                    value=self.winsor,
                                                    type="number",
                                                    min=0,
                                                    max=49,
                                                    step=1,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        hide=self.hide_winsor,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Row(
                                                            [
                                                                dbc.Label("Dispersão:"),
                                                                dbc.Tooltip(
                                                                    "Para características categóricas, exibir "
                                                                    "uma nuvem de pontos ao lado dos gráficos de violino.",
                                                                    target="reg-vs-col-show-points-"
                                                                    + self.name,
                                                                ),
                                                                dbc.Checklist(
                                                                    options=[
                                                                        {
                                                                            "label": "Mostrar nuvem de pontos",
                                                                            "value": True,
                                                                        }
                                                                    ],
                                                                    value=[True]
                                                                    if self.points
                                                                    else [],
                                                                    id="reg-vs-col-show-points-"
                                                                    + self.name,
                                                                    inline=True,
                                                                    switch=True,
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    id="reg-vs-col-show-points-div-"
                                                    + self.name,
                                                )
                                            ],
                                            md=2,
                                        ),
                                        self.hide_points,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        dbc.Label(
                                                            "Categorias:",
                                                            id="reg-vs-col-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Número máximo de categorias a exibir",
                                                            target="reg-vs-col-n-categories-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Input(
                                                            id="reg-vs-col-n-categories-"
                                                            + self.name,
                                                            value=self.cats_topx,
                                                            type="number",
                                                            min=1,
                                                            max=50,
                                                            step=1,
                                                        ),
                                                    ],
                                                    id="reg-vs-col-n-categories-div-"
                                                    + self.name,
                                                ),
                                            ],
                                            md=2,
                                        ),
                                        self.hide_cats_topx,
                                    ),
                                    make_hideable(
                                        dbc.Col(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label(
                                                            "Ordenar categorias:",
                                                            id="reg-vs-col-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Tooltip(
                                                            "Como ordenar as categorias: Alfabeticamente, mais comuns "
                                                            "primeiro (Frequência), ou maior valor médio absoluto de SHAP primeiro (Impacto Shap)",
                                                            target="reg-vs-col-categories-sort-label-"
                                                            + self.name,
                                                        ),
                                                        dbc.Select(
                                                            id="reg-vs-col-categories-sort-"
                                                            + self.name,
                                                            options=[
                                                                {
                                                                    "label": "Alfabeticamente",
                                                                    "value": "alphabet",
                                                                },
                                                                {
                                                                    "label": "Frequência",
                                                                    "value": "freq",
                                                                },
                                                                {
                                                                    "label": "Impacto Shap",
                                                                    "value": "shap",
                                                                },
                                                            ],
                                                            value=self.cats_sort,
                                                        ),
                                                    ],
                                                    id="reg-vs-col-categories-sort-div-"
                                                    + self.name,
                                                ),
                                            ],
                                            md=4,
                                        ),
                                        hide=self.hide_cats_sort,
                                    ),
                                ]
                            )
                        ]
                    ),
                    hide=self.hide_footer,
                ),
            ]
        )

    def to_html(self, state_dict=None, add_header=True):
        args = self.get_state_args(state_dict)
        if args["display"] == "observed":
            fig = self.explainer.plot_y_vs_feature(
                args["col"],
                points=bool(args["points"]),
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        elif args["display"] == "predicted":
            fig = self.explainer.plot_preds_vs_feature(
                args["col"],
                points=bool(args["points"]),
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        else:
            fig = self.explainer.plot_residuals_vs_feature(
                args["col"],
                points=bool(args["points"]),
                winsor=args["winsor"],
                dropna=True,
                topx=args["cats_topx"],
                sort=args["cats_sort"],
                round=self.round,
                plot_sample=self.plot_sample,
            )
        fig.update_layout(margin=dict(t=40, b=40, l=40, r=40))
        html = to_html.card(to_html.fig(fig), title=self.title, subtitle=self.subtitle)
        if add_header:
            return to_html.add_header(html)
        return html

    def component_callbacks(self, app):
        @app.callback(
            [
                Output("reg-vs-col-graph-" + self.name, "figure"),
                Output("reg-vs-col-show-points-div-" + self.name, "style"),
                Output("reg-vs-col-n-categories-div-" + self.name, "style"),
                Output("reg-vs-col-categories-sort-div-" + self.name, "style"),
            ],
            [
                Input("reg-vs-col-col-" + self.name, "value"),
                Input("reg-vs-col-display-type-" + self.name, "value"),
                Input("reg-vs-col-show-points-" + self.name, "value"),
                Input("reg-vs-col-winsor-" + self.name, "value"),
                Input("reg-vs-col-n-categories-" + self.name, "value"),
                Input("reg-vs-col-categories-sort-" + self.name, "value"),
            ],
        )
        def update_residuals_graph(col, display, points, winsor, topx, sort):
            if (
                col in self.explainer.onehot_cols
                or col in self.explainer.categorical_cols
            ):
                style = {}
            else:
                style = dict(display="none")
            if display == "observed":
                return (
                    self.explainer.plot_y_vs_feature(
                        col,
                        points=bool(points),
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    ),
                    style,
                    style,
                    style,
                )
            elif display == "predicted":
                return (
                    self.explainer.plot_preds_vs_feature(
                        col,
                        points=bool(points),
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    ),
                    style,
                    style,
                    style,
                )
            else:
                return (
                    self.explainer.plot_residuals_vs_feature(
                        col,
                        residuals=display,
                        points=bool(points),
                        winsor=winsor,
                        dropna=True,
                        topx=topx,
                        sort=sort,
                        round=self.round,
                        plot_sample=self.plot_sample,
                    ),
                    style,
                    style,
                    style,
                )