"""Add user_id column to answer_data table

Revision ID: d2b4d5a7859a
Revises: 
Create Date: 2024-04-12 11:53:12.042027

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'd2b4d5a7859a'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('form_data')
    op.drop_table('query_data')
    op.drop_table('admin')
    with op.batch_alter_table('answer_data', schema=None) as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer(), nullable=False))
        batch_op.add_column(sa.Column('question', sa.String(length=255), nullable=False))
        batch_op.add_column(sa.Column('context', sa.Text(), nullable=False))
        batch_op.add_column(sa.Column('answer', sa.Text(), nullable=True))
        batch_op.create_foreign_key('fk_answer_data_user_id', 'user', ['user_id'], ['id'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('answer_data', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_column('answer')
        batch_op.drop_column('context')
        batch_op.drop_column('question')
        batch_op.drop_column('user_id')

    op.create_table('admin',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('username', sa.VARCHAR(length=50), nullable=False),
    sa.Column('password', sa.VARCHAR(length=50), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('username')
    )
    op.create_table('query_data',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('user_id', sa.INTEGER(), nullable=False),
    sa.Column('question', sa.VARCHAR(length=255), nullable=False),
    sa.Column('context', sa.TEXT(), nullable=False),
    sa.Column('answer', sa.TEXT(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('form_data',
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('user_id', sa.INTEGER(), nullable=False),
    sa.Column('question', sa.VARCHAR(length=255), nullable=False),
    sa.Column('context', sa.TEXT(), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###