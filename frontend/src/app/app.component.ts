import { Component } from '@angular/core';
import {TextService} from './text.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'client';
  playerName;
  responseText;
  constructor(private textService: TextService) {
  this.playerName = '';
  this.responseText = '';
  }

  submitText() {
    console.log('Sent: ' + this.playerName);
    this.textService.submitText(this.playerName).subscribe((response: any) => {
      console.log('Received: ' + response);
      this.responseText = response.response;
    });
  }

}
