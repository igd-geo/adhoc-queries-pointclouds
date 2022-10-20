use std::ops::Range;

use pasture_core::nalgebra::Vector3;

use super::Block;

#[derive(Debug, Copy, Clone)]
pub enum IndexResult {
    MatchAll,
    MatchSome,
    NoMatch,
}

impl IndexResult {
    pub fn and(self, other: Self) -> Self {
        match (self, other) {
            (IndexResult::MatchAll, IndexResult::MatchAll) => IndexResult::MatchAll,
            (IndexResult::MatchAll, IndexResult::MatchSome) => IndexResult::MatchSome,
            (IndexResult::MatchSome, IndexResult::MatchAll) => IndexResult::MatchSome,
            (IndexResult::MatchSome, IndexResult::MatchSome) => IndexResult::MatchSome,
            (IndexResult::NoMatch, _) => IndexResult::NoMatch,
            (_, IndexResult::NoMatch) => IndexResult::NoMatch,
        }
    }

    pub fn or(self, other: Self) -> Self {
        match (self, other) {
            (IndexResult::MatchAll, _) => IndexResult::MatchAll,
            (_, IndexResult::MatchAll) => IndexResult::MatchAll,
            (IndexResult::MatchSome, _) => IndexResult::MatchSome,
            (_, IndexResult::MatchSome) => IndexResult::MatchSome,
            _ => IndexResult::NoMatch,
        }
    }
}

pub trait Index: Send + Sync {
    fn within(&self, range: &Range<Value>, num_points_in_block: usize) -> IndexResult;
    fn equals(&self, data: &Value, num_points_in_block: usize) -> IndexResult;
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Classification,
    Position3D,
}

#[derive(Copy, Clone, Debug)]
pub struct Position(pub Vector3<f64>);
#[derive(Copy, Clone, Debug)]
pub struct Classification(pub u8);

pub enum Value {
    Classification(Classification),
    Position(Position),
}

impl Value {
    pub fn value_type(&self) -> &ValueType {
        match self {
            Value::Classification(_) => &ValueType::Classification,
            Value::Position(_) => &ValueType::Position3D,
        }
    }
}

pub enum QueryExpression {
    Within(Range<Value>),
    Equals(Value),
    And(Box<QueryExpression>, Box<QueryExpression>),
    Or(Box<QueryExpression>, Box<QueryExpression>),
}

impl QueryExpression {
    pub fn eval(&self, block: &Block) -> IndexResult {
        match self {
            QueryExpression::Within(range) => {
                if let Some(index) = block.indices().get(range.start.value_type()) {
                    index.within(range, block.len())
                } else {
                    // Without an index, we can't rule out the block
                    IndexResult::MatchSome
                }
            }
            QueryExpression::Equals(expr) => {
                if let Some(index) = block.indices().get(expr.value_type()) {
                    index.equals(expr, block.len())
                } else {
                    IndexResult::MatchSome
                }
            }
            QueryExpression::And(l_expr, r_expr) => l_expr.eval(block).and(r_expr.eval(block)),
            QueryExpression::Or(l_expr, r_expr) => l_expr.eval(block).or(r_expr.eval(block)),
        }
    }
}

pub type Query = QueryExpression;
