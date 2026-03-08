import streamlit as st
from typing import Optional


def render_section_header(title: str, subtitle: Optional[str] = None) -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f"<p class='pc-subtitle'>{subtitle}</p>", unsafe_allow_html=True)


def render_metric_cards(metrics: dict) -> None:
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f"""
                <div class="pc-card pc-metric-card">
                    <div class="pc-metric-label">{label}</div>
                    <div class="pc-metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def card(title: Optional[str] = None):
    class CardContext:
        def __enter__(self):
            st.markdown("<div class='pc-card'>", unsafe_allow_html=True)
            if title:
                st.markdown(f"<div class='pc-card-title'>{title}</div>", unsafe_allow_html=True)
            return self

        def __exit__(self, exc_type, exc, exc_tb):
            st.markdown("</div>", unsafe_allow_html=True)

    return CardContext()

