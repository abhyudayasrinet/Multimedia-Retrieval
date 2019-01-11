import React, { Component } from 'react';
import ListItem from '@material-ui/core/ListItem';
import ListItemIcon from '@material-ui/core/ListItemIcon';
import ListItemText from '@material-ui/core/ListItemText';
import AssessmentIcon from '@material-ui/icons/Assessment';

class TaskDrawer extends Component {
  state = {
    tasks: [],
  }

  componentWillReceiveProps(nextProps) {
    this.setState({tasks: nextProps.tasks});
  }

  render() {
    const tasks = this.state.tasks
      .map((task, i) =>
      <div key={i}>
        <ListItem button onClick={() => this.props.selectedTaskHandler(i)}>
          <ListItemIcon>
            <AssessmentIcon />
          </ListItemIcon>
          <ListItemText>
            Task {task.task} ( {task.algo} )
          </ListItemText>
        </ListItem>
      </div>
    );

    return (
      <div>
        {tasks}
      </div>
    );
  }
}

export default TaskDrawer;
